import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/notification_repository.dart';
import '../../data/mock_notification_repository.dart';
import '../../domain/models/app_notification.dart';
import '../../domain/states/notification_state.dart';

/// Overrideable provider for the notification repository.
/// Defaults to [MockNotificationRepository] for development.
final notificationRepositoryProvider = Provider<NotificationRepository>((ref) {
  return MockNotificationRepository();
});

/// Main notification state provider.
final notificationProvider =
    StateNotifierProvider<NotificationNotifier, NotificationState>((ref) {
  final repo = ref.watch(notificationRepositoryProvider);
  return NotificationNotifier(repo);
});

/// Derived provider for the unread count — used by [UnreadBadge].
/// Single source of truth across all screens.
final unreadCountProvider = Provider<int>((ref) {
  final state = ref.watch(notificationProvider);
  return state.maybeWhen(
    loaded: (_, unreadCount) => unreadCount,
    orElse: () => 0,
  );
});

/// Manages notification state: loading, CRUD, and incoming push handling.
class NotificationNotifier extends StateNotifier<NotificationState> {
  final NotificationRepository _repo;

  NotificationNotifier(this._repo) : super(const NotificationState.initial());

  /// Load all notifications from the repository.
  Future<void> loadNotifications() async {
    state = const NotificationState.loading();
    try {
      final notifications = await _repo.loadNotifications();
      final unread = notifications.where((n) => !n.isRead).length;
      state = NotificationState.loaded(
        notifications: notifications,
        unreadCount: unread,
      );
    } catch (e) {
      state = NotificationState.error(e.toString());
    }
  }

  /// Mark a single notification as read.
  Future<void> markRead(String id) async {
    try {
      await _repo.markRead(id);
      state.maybeWhen(
        loaded: (notifications, unreadCount) {
          final updated = notifications.map((n) {
            if (n.id == id && !n.isRead) {
              return n.copyWith(isRead: true);
            }
            return n;
          }).toList();
          final newUnread = updated.where((n) => !n.isRead).length;
          state = NotificationState.loaded(
            notifications: updated,
            unreadCount: newUnread,
          );
        },
        orElse: () {},
      );
    } catch (e) {
      // Silently fail — don't disrupt UI for a read-status update.
    }
  }

  /// Mark all notifications as read.
  Future<void> markAllRead() async {
    try {
      await _repo.markAllRead();
      state.maybeWhen(
        loaded: (notifications, _) {
          final updated =
              notifications.map((n) => n.copyWith(isRead: true)).toList();
          state = NotificationState.loaded(
            notifications: updated,
            unreadCount: 0,
          );
        },
        orElse: () {},
      );
    } catch (e) {
      // Silently fail.
    }
  }

  /// Add an incoming notification (from FCM or STOMP).
  /// Prepends to the list and increments unread count.
  void addIncoming(AppNotification notification) {
    state.maybeWhen(
      loaded: (notifications, unreadCount) {
        state = NotificationState.loaded(
          notifications: [notification, ...notifications],
          unreadCount: unreadCount + 1,
        );
      },
      orElse: () {
        // If not yet loaded, just set a loaded state with this one item.
        state = NotificationState.loaded(
          notifications: [notification],
          unreadCount: 1,
        );
      },
    );
  }
}
