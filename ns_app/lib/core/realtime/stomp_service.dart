import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:rxdart/rxdart.dart';
import 'package:stomp_dart_client/stomp.dart';
import 'package:stomp_dart_client/stomp_config.dart';
import 'package:stomp_dart_client/stomp_frame.dart';
import 'stomp_state.dart';
import 'stock_update_message.dart';

final stompServiceProvider = Provider<StompService>((ref) {
  final service = StompService(ref);
  ref.onDispose(() => service.dispose());
  return service;
});

final stompStateProvider = StreamProvider<StompState>((ref) {
  return ref.watch(stompServiceProvider).stateStream;
});

/// Singleton STOMP-over-WebSocket client.
///
/// Handles connection lifecycle, automatic reconnection, and
/// subscription management for stock updates and user notifications.
class StompService {
  final Ref ref;
  StompClient? _client;
  String? _accessToken;

  final BehaviorSubject<StompState> _stateController =
      BehaviorSubject.seeded(const StompState.disconnected());

  /// Active STOMP unsubscribe callbacks keyed by destination.
  final Map<String, dynamic Function({Map<String, String>? unsubscribeHeaders})>
      _subscriptions = {};

  /// Stored callback references so we can re-subscribe after reconnect.
  final Map<String, void Function(StockUpdateMessage)> _retailerCallbacks = {};

  /// Set of retailer IDs we're tracking — survives reconnect.
  final Set<String> _activeRetailerIds = {};

  /// User ID for private notification queue.
  String? _userId;

  /// Called when a notification arrives over the WebSocket channel.
  void Function(Map<String, dynamic> payload)? onNotificationReceived;

  StompService(this.ref);

  Stream<StompState> get stateStream => _stateController.stream;
  StompState get currentState => _stateController.value;
  bool get isConnected => currentState.maybeWhen(
        connected: (_) => true,
        orElse: () => false,
      );

  // ---------------------------------------------------------------------------
  // Connection
  // ---------------------------------------------------------------------------

  void connect(String accessToken, {String? userId}) {
    _accessToken = accessToken;
    _userId = userId;

    _client?.deactivate();

    _client = StompClient(
      config: StompConfig(
        url: _getWsUrl(),
        onConnect: _onConnected,
        onDisconnect: _onDisconnected,
        onStompError: (frame) => _onError(frame, null),
        onWebSocketError: (error) => _onError(null, error),
        connectionTimeout: const Duration(seconds: 10),
        reconnectDelay: const Duration(seconds: 5),
        stompConnectHeaders: {
          'Authorization': 'Bearer $accessToken',
        },
        webSocketConnectHeaders: {
          'Authorization': 'Bearer $accessToken',
        },
      ),
    );

    _client!.activate();
    _stateController.add(const StompState.connecting());
  }

  void disconnect() {
    for (final unsub in _subscriptions.values) {
      try {
        unsub();
      } catch (_) {}
    }
    _subscriptions.clear();
    _retailerCallbacks.clear();
    _activeRetailerIds.clear();
    _userId = null;
    _client?.deactivate();
    _client = null;
    _stateController.add(const StompState.disconnected());
  }

  // ---------------------------------------------------------------------------
  // STOMP callbacks
  // ---------------------------------------------------------------------------

  void _onConnected(StompFrame frame) {
    _stateController
        .add(StompState.connected(frame.headers['session'] ?? ''));

    // Re-subscribe to user notification queue
    if (_userId != null) {
      _subscribeToUserNotificationsInternal(_userId!);
    }

    // Re-subscribe to all active retailer topics
    final retailers = Set<String>.from(_activeRetailerIds);
    _subscriptions.clear(); // old subs are dead after reconnect
    for (final retailerId in retailers) {
      final callback = _retailerCallbacks[retailerId];
      if (callback != null) {
        _subscribeToRetailerInternal(retailerId, callback);
      }
    }

    debugPrint('[STOMP] Connected. Re-subscribed to ${retailers.length} '
        'retailer(s).');
  }

  void _onDisconnected(StompFrame frame) {
    // Don't clear _activeRetailerIds or _retailerCallbacks —
    // we need them for reconnect.
    _subscriptions.clear();
    _stateController.add(const StompState.disconnected());
    debugPrint('[STOMP] Disconnected. Will auto-reconnect.');
  }

  void _onError(StompFrame? frame, dynamic error) {
    final message = error?.toString() ?? frame?.body ?? 'Unknown STOMP error';
    _stateController.add(StompState.error(message));
    debugPrint('[STOMP] Error: $message');
  }

  // ---------------------------------------------------------------------------
  // Retailer stock subscriptions
  // ---------------------------------------------------------------------------

  void subscribeToRetailer(
    String retailerId,
    void Function(StockUpdateMessage) onMessage,
  ) {
    if (_activeRetailerIds.contains(retailerId)) return;
    _activeRetailerIds.add(retailerId);
    _retailerCallbacks[retailerId] = onMessage;

    if (isConnected) {
      _subscribeToRetailerInternal(retailerId, onMessage);
    }
    // If not connected, will be subscribed on reconnect via _onConnected.
  }

  void _subscribeToRetailerInternal(
    String retailerId,
    void Function(StockUpdateMessage) onMessage,
  ) {
    final destination = '/topic/stock/$retailerId';
    if (_subscriptions.containsKey(destination)) return;

    final unsub = _client!.subscribe(
      destination: destination,
      callback: (frame) {
        if (frame.body == null) return;
        try {
          final msg = StockUpdateMessage.fromJson(
            jsonDecode(frame.body!) as Map<String, dynamic>,
          );
          onMessage(msg);
        } catch (e) {
          debugPrint('[STOMP] Failed to parse stock update: $e');
        }
      },
    );
    _subscriptions[destination] = unsub;
  }

  void unsubscribeFromRetailer(String retailerId) {
    final destination = '/topic/stock/$retailerId';
    try {
      _subscriptions[destination]?.call();
    } catch (_) {}
    _subscriptions.remove(destination);
    _activeRetailerIds.remove(retailerId);
    _retailerCallbacks.remove(retailerId);
  }

  // ---------------------------------------------------------------------------
  // User notification queue
  // ---------------------------------------------------------------------------

  void subscribeToUserNotifications(String userId) {
    _userId = userId;
    if (isConnected) {
      _subscribeToUserNotificationsInternal(userId);
    }
  }

  void _subscribeToUserNotificationsInternal(String userId) {
    final destination = '/user/$userId/queue/notifications';
    if (_subscriptions.containsKey(destination)) return;

    final unsub = _client!.subscribe(
      destination: destination,
      callback: (frame) {
        if (frame.body == null) return;
        try {
          final payload =
              jsonDecode(frame.body!) as Map<String, dynamic>;
          onNotificationReceived?.call(payload);
        } catch (e) {
          debugPrint('[STOMP] Failed to parse notification: $e');
        }
      },
    );
    _subscriptions[destination] = unsub;
  }

  // ---------------------------------------------------------------------------
  // Retry
  // ---------------------------------------------------------------------------

  void retry() {
    if (_accessToken != null) {
      connect(_accessToken!, userId: _userId);
    }
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  String _getWsUrl() {
    // TODO: Replace with ApiConstants.wsBaseUrl when available
    return 'ws://localhost:8080/ws';
  }

  void dispose() {
    disconnect();
    _stateController.close();
  }
}

/// Typedef exposed for typing convenience outside this file.
typedef StompStateConnected = StompState;
