import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'notification_router.dart';

/// Manages the flutter_local_notifications plugin for showing
/// notification UI when the app is in the foreground.
class LocalNotificationService {
  static final LocalNotificationService _instance =
      LocalNotificationService._internal();
  factory LocalNotificationService() => _instance;
  LocalNotificationService._internal();

  final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();

  bool _initialised = false;

  Future<void> init() async {
    if (_initialised) return;

    const AndroidInitializationSettings androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    const DarwinInitializationSettings iosSettings =
        DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );

    const InitializationSettings settings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _plugin.initialize(
      settings,
      onDidReceiveNotificationResponse: _onNotificationTapped,
    );

    // Create the high-importance Android notification channel.
    const AndroidNotificationChannel channel = AndroidNotificationChannel(
      'stock_updates',
      'Stock Updates',
      importance: Importance.high,
      description: 'Notifications for stock updates and deals',
    );

    await _plugin
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(channel);

    _initialised = true;
    debugPrint('[LocalNotification] Initialised.');
  }

  /// Handle notification tap — parse payload and route.
  static void _onNotificationTapped(NotificationResponse response) {
    if (response.payload != null && response.payload!.isNotEmpty) {
      NotificationRouter.handlePayload(response.payload!);
    }
  }

  /// Show a local notification from a [RemoteMessage].
  Future<void> showNotification(RemoteMessage message) async {
    final notification = message.notification;
    if (notification == null) return;

    final String bodyText = notification.body ?? '';

    final AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
      'stock_updates',
      'Stock Updates',
      importance: Importance.max,
      priority: Priority.high,
      styleInformation: BigTextStyleInformation(bodyText),
    );

    const DarwinNotificationDetails iosDetails = DarwinNotificationDetails();

    final NotificationDetails details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _plugin.show(
      notification.hashCode,
      notification.title,
      bodyText,
      details,
      payload: jsonEncode(message.data),
    );
  }

  /// Show a local notification from raw title/body/data (for STOMP messages).
  Future<void> showRawNotification({
    required String title,
    required String body,
    Map<String, dynamic>? data,
  }) async {
    final AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
      'stock_updates',
      'Stock Updates',
      importance: Importance.max,
      priority: Priority.high,
      styleInformation: BigTextStyleInformation(body),
    );

    const DarwinNotificationDetails iosDetails = DarwinNotificationDetails();

    final NotificationDetails details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _plugin.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      details,
      payload: data != null ? jsonEncode(data) : null,
    );
  }
}
