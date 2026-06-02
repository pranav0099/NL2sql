import '../domain/models/app_notification.dart';

abstract class NotificationRepository {
  Future<List<AppNotification>> loadNotifications();
  Future<void> markRead(String id);
  Future<void> markAllRead();
  Future<int> getUnreadCount();
  Future<void> registerFcmToken(String token);
}
