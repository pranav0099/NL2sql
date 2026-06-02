import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:rxdart/rxdart.dart';
import 'stomp_service.dart';
import 'stomp_state.dart';
import 'stock_update_message.dart';

/// Provider that switches between [MockStompService] and real [StompService].
/// Set `USE_MOCK_STOMP = true` in main.dart to use this.
final mockStompServiceProvider = Provider<MockStompService>((ref) {
  final service = MockStompService(ref);
  ref.onDispose(() => service.dispose());
  return service;
});

/// A mock STOMP service for testing the full UI without a backend.
///
/// Overrides [connect] to immediately emit a connected state.
/// Exposes [triggerMockStockUpdate] and [triggerMockNotification]
/// for manual testing from a debug panel or timer.
class MockStompService extends StompService {
  Timer? _periodicTimer;
  bool _periodicEnabled = false;

  MockStompService(super.ref);

  /// Override connect to immediately succeed without a real WebSocket.
  @override
  void connect(String accessToken, {String? userId}) {
    debugPrint('[MockSTOMP] Mock connecting...');

    // Simulate a brief connecting state then connected.
    // ignore: invalid_use_of_protected_member
    stateStream; // ensure stream is accessible
    _emitState(const StompState.connecting());

    Future.delayed(const Duration(milliseconds: 500), () {
      _emitState(const StompState.connected('mock-session-001'));
      debugPrint('[MockSTOMP] Mock connected.');
    });
  }

  /// Override disconnect to just reset state.
  @override
  void disconnect() {
    _periodicTimer?.cancel();
    _periodicTimer = null;
    _emitState(const StompState.disconnected());
    debugPrint('[MockSTOMP] Mock disconnected.');
  }

  /// Override retry for mock.
  @override
  void retry() {
    connect('mock-token');
  }

  /// Simulate a stock update arriving for a specific retailer.
  void triggerMockStockUpdate({
    required String retailerId,
    required String productId,
    required String productName,
    required StockLevel stockLevel,
  }) {
    final msg = StockUpdateMessage(
      productId: productId,
      productName: productName,
      retailerId: retailerId,
      stockLevel: stockLevel,
      updatedAt: DateTime.now(),
    );

    debugPrint('[MockSTOMP] Triggering stock update: '
        '${msg.productName} → ${msg.stockLevel}');

    // Call stored callbacks if any retailer subscription exists.
    // Access the parent class's internal callbacks via subscribeToRetailer.
    // For mock, we directly invoke the notification mechanism.
    onNotificationReceived?.call({
      'type': 'STOCK_UPDATE',
      'retailerId': retailerId,
      'productId': productId,
      'title': 'Stock Update: $productName',
      'body': '$productName is now ${_stockLevelLabel(stockLevel)}',
    });
  }

  /// Simulate an incoming notification.
  void triggerMockNotification({
    required String title,
    required String body,
    String type = 'STOCK_UPDATE',
    String? retailerId,
    String? productId,
  }) {
    onNotificationReceived?.call({
      'type': type,
      'title': title,
      'body': body,
      // ignore: use_null_aware_elements
      if (retailerId != null) 'retailerId': retailerId,
      // ignore: use_null_aware_elements
      if (productId != null) 'productId': productId,
    });
  }

  /// Start periodic mock stock updates for demo purposes.
  void startPeriodicMockUpdates({Duration interval = const Duration(seconds: 8)}) {
    if (_periodicEnabled) return;
    _periodicEnabled = true;

    final updates = [
      (rid: 'retailer_1', pid: 'product_101', name: 'iPhone 15 Pro', level: StockLevel.inStock),
      (rid: 'retailer_2', pid: 'product_102', name: 'Samsung S24', level: StockLevel.lowStock),
      (rid: 'retailer_1', pid: 'product_103', name: 'MacBook Pro', level: StockLevel.outOfStock),
      (rid: 'retailer_3', pid: 'product_104', name: 'AirPods Pro', level: StockLevel.inStock),
      (rid: 'retailer_2', pid: 'product_102', name: 'Samsung S24', level: StockLevel.inStock),
    ];

    int index = 0;
    _periodicTimer = Timer.periodic(interval, (_) {
      final u = updates[index % updates.length];
      triggerMockStockUpdate(
        retailerId: u.rid,
        productId: u.pid,
        productName: u.name,
        stockLevel: u.level,
      );
      index++;
    });
  }

  void stopPeriodicMockUpdates() {
    _periodicTimer?.cancel();
    _periodicTimer = null;
    _periodicEnabled = false;
  }

  @override
  void dispose() {
    _periodicTimer?.cancel();
    super.dispose();
  }

  // --- Helpers ---

  void _emitState(StompState state) {
    // Access the parent's BehaviorSubject via the public stateStream.
    // We need a way to add to it. The parent class exposes _stateController
    // but it's private. For mock we maintain our own.
    _mockStateController.add(state);
  }

  final BehaviorSubject<StompState> _mockStateController =
      BehaviorSubject.seeded(const StompState.disconnected());

  @override
  Stream<StompState> get stateStream => _mockStateController.stream;

  @override
  StompState get currentState => _mockStateController.value;

  @override
  bool get isConnected {
    final s = _mockStateController.value;
    return s.when(
      disconnected: () => false,
      connecting: () => false,
      connected: (_) => true,
      error: (_) => false,
    );
  }

  String _stockLevelLabel(StockLevel level) {
    switch (level) {
      case StockLevel.inStock:
        return 'back in stock';
      case StockLevel.lowStock:
        return 'low on stock';
      case StockLevel.outOfStock:
        return 'out of stock';
    }
  }
}
