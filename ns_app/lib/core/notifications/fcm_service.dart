import 'package:flutter/widgets.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'local_notification_service.dart';
import 'notification_router.dart';
import '../../features/notifications/data/notification_repository.dart';
import '../../features/notifications/domain/models/app_notification.dart';
import '../../features/notifications/presentation/providers/notification_provider.dart';

/// Top-level provider for the notification repository.
/// Override this in ProviderScope with mock or HTTP implementation.
final notificationRepositoryProvider = Provider<NotificationRepository>((ref) {
  throw UnimplementedError(
    'notificationRepositoryProvider must be overridden '
    'in ProviderScope with a mock or HTTP implementation.',
  );
});

final fcmServiceProvider = Provider<FcmService>((ref) {
  return FcmService(ref);
});

/// Handles FCM initialisation, token management, and message routing.
///
/// Works in all three app states:
/// - **Foreground**: shows local notification + updates provider
/// - **Background**: FCM auto-shows notification; tap handled by onMessageOpenedApp
/// - **Terminated**: getInitialMessage handles cold-start from notification tap
class FcmService with WidgetsBindingObserver {
  final Ref ref;
  final FirebaseMessaging _messaging = FirebaseMessaging.instance;

  FcmService(this.ref);

  Future<void> init({required bool isAuthenticated}) async {
    WidgetsBinding.instance.addObserver(this);

    if (!isAuthenticated) return;

    // Request notification permission (primarily for iOS).
    final NotificationSettings settings = await _messaging.requestPermission(
      alert: true,
      badge: true,
      sound: true,
    );

    if (settings.authorizationStatus == AuthorizationStatus.authorized ||
        settings.authorizationStatus == AuthorizationStatus.provisional) {
      // Register FCM token with backend (idempotent PUT).
      String? token = await _messaging.getToken();
      if (token != null) {
        await _registerTokenWithBackend(token);
      }

      // Listen for token refresh.
      _messaging.onTokenRefresh.listen((newToken) {
        _registerTokenWithBackend(newToken);
      });
    }

    // ---- Foreground messages ----
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      debugPrint('[FCM] Foreground message: ${message.messageId}');
      // Show local notification UI.
      LocalNotificationService().showNotification(message);
      // Update in-app notification state.
      _handleIncomingNotification(message);
    });

    // ---- Background tap (app was backgrounded) ----
    FirebaseMessaging.onMessageOpenedApp.listen((message) {
      debugPrint('[FCM] Opened from background: ${message.messageId}');
      NotificationRouter.route(message.data);
    });

    // ---- Terminated tap (cold start from notification) ----
    final RemoteMessage? initial = await _messaging.getInitialMessage();
    if (initial != null) {
      debugPrint('[FCM] Opened from terminated: ${initial.messageId}');
      // Slight delay to let router initialise.
      Future.delayed(const Duration(milliseconds: 500), () {
        NotificationRouter.route(initial.data);
      });
    }
  }





  /// Parse the FCM data payload into an [AppNotification] and
  /// push it into the notification provider.
  void _handleIncomingNotification(RemoteMessage message) {
    try {
      final data = message.data;
      final notification = AppNotification(
        id: message.messageId ?? DateTime.now().toIso8601String(),
        type: _parseNotificationType(data['type']),
        title: message.notification?.title ?? data['title'] ?? 'Notification',
        body: message.notification?.body ?? data['body'] ?? '',
        retailerId: data['retailerId'],
        productId: data['productId'],
        sentAt: DateTime.now(),
        isRead: false,
      );
      ref.read(notificationProvider.notifier).addIncoming(notification);
    } catch (e) {
      debugPrint('[FCM] Failed to handle incoming notification: $e');
    }
  }

  NotificationType _parseNotificationType(String? type) {
    switch (type) {
      case 'STOCK_UPDATE':
        return NotificationType.stockUpdate;
      case 'DEAL':
        return NotificationType.deal;
      default:
        return NotificationType.stockUpdate;
    }
  }

  Future<void> _registerTokenWithBackend(String token) async {
    try {
      await ref.read(notificationRepositoryProvider).registerFcmToken(token);
      debugPrint('[FCM] Token registered with backend.');
    } catch (e) {
      debugPrint('[FCM] Failed to register token: $e');
    }
  }

  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
  }
}
