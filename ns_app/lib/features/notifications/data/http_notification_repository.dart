import 'package:dio/dio.dart';
import '../domain/models/app_notification.dart';
import 'notification_repository.dart';

class HttpNotificationRepository implements NotificationRepository {
  final Dio _dio;

  HttpNotificationRepository(this._dio);

  @override
  Future<List<AppNotification>> loadNotifications() async {
    final response = await _dio.get('/api/v1/notifications');
    return (response.data as List)
        .map((json) => AppNotification.fromJson(json))
        .toList();
  }

  @override
  Future<void> markRead(String id) async {
    await _dio.put('/api/v1/notifications/$id/read');
  }

  @override
  Future<void> markAllRead() async {
    await _dio.put('/api/v1/notifications/read-all');
  }

  @override
  Future<int> getUnreadCount() async {
    final response = await _dio.get('/api/v1/notifications/unread-count');
    return response.data['count'] as int;
  }

  @override
  Future<void> registerFcmToken(String token) async {
    await _dio.put('/api/v1/customer/fcm-token', data: {'fcmToken': token});
  }
}
