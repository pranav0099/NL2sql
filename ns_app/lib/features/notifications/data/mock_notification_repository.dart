import 'dart:math';
import '../domain/models/app_notification.dart';
import 'notification_repository.dart';

class MockNotificationRepository implements NotificationRepository {
  @override
  Future<List<AppNotification>> loadNotifications() async {
    await Future.delayed(const Duration(milliseconds: 600));
    return [
      AppNotification(
        id: '1',
        type: NotificationType.stockUpdate,
        title: 'Product Back in Stock',
        body: 'iPhone 15 Pro is now available at Tech Store',
        retailerId: 'retailer_1',
        productId: 'product_101',
        sentAt: DateTime.now().subtract(const Duration(hours: 2)),
        isRead: true,
      ),
      AppNotification(
        id: '2',
        type: NotificationType.stockUpdate,
        title: 'Low Stock Alert',
        body: 'Samsung S24 has only 3 units left',
        retailerId: 'retailer_2',
        productId: 'product_102',
        sentAt: DateTime.now().subtract(const Duration(hours: 5)),
        isRead: true,
      ),
      AppNotification(
        id: '3',
        type: NotificationType.stockUpdate,
        title: 'Out of Stock',
        body: 'MacBook Pro is currently out of stock',
        retailerId: 'retailer_1',
        productId: 'product_103',
        sentAt: DateTime.now().subtract(const Duration(minutes: 30)),
        isRead: false,
      ),
      AppNotification(
        id: '4',
        type: NotificationType.deal,
        title: 'Special Deal',
        body: '20% off on all accessories today only!',
        retailerId: 'retailer_3',
        sentAt: DateTime.now().subtract(const Duration(hours: 1)),
        isRead: false,
      ),
      AppNotification(
        id: '5',
        type: NotificationType.deal,
        title: 'Flash Sale',
        body: 'Flash sale ends in 2 hours - don\'t miss out!',
        retailerId: 'retailer_2',
        sentAt: DateTime.now().subtract(const Duration(minutes: 45)),
        isRead: false,
      ),
    ];
  }

  @override
  Future<void> markRead(String id) async {
    await Future.delayed(const Duration(milliseconds: 200));
  }

  @override
  Future<void> markAllRead() async {
    await Future.delayed(const Duration(milliseconds: 400));
  }

  @override
  Future<int> getUnreadCount() async {
    await Future.delayed(const Duration(milliseconds: 200));
    return 3 + Random().nextInt(5);
  }

  @override
  Future<void> registerFcmToken(String token) async {
    // Mock - no-op
  }
}
