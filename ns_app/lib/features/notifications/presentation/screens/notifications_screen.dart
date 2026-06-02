import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/notification_provider.dart';
import '../widgets/notification_tile.dart';
import '../widgets/realtime_status_banner.dart';

/// Full notification inbox screen.
///
/// - AppBar with "Notifications" title + "Mark all read" action
/// - Pull-to-refresh
/// - Loading / error / empty states
/// - Auto marks-all-read after 2 second delay on screen enter
class NotificationsScreen extends ConsumerStatefulWidget {
  const NotificationsScreen({super.key});

  @override
  ConsumerState<NotificationsScreen> createState() =>
      _NotificationsScreenState();
}

class _NotificationsScreenState extends ConsumerState<NotificationsScreen> {
  Timer? _markAllReadTimer;

  @override
  void initState() {
    super.initState();
    // Load notifications on screen enter.
    Future.microtask(() {
      ref.read(notificationProvider.notifier).loadNotifications();
    });

    // Auto mark-all-read after 2 seconds so user sees unread state briefly.
    _markAllReadTimer = Timer(const Duration(seconds: 2), () {
      if (mounted) {
        ref.read(notificationProvider.notifier).markAllRead();
      }
    });
  }

  @override
  void dispose() {
    _markAllReadTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final state = ref.watch(notificationProvider);

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      appBar: AppBar(
        title: const Text('Notifications'),
        centerTitle: false,
        actions: [
          TextButton.icon(
            onPressed: () {
              ref.read(notificationProvider.notifier).markAllRead();
            },
            icon: Icon(
              Icons.done_all_rounded,
              size: 18,
              color: theme.colorScheme.primary,
            ),
            label: Text(
              'Mark all read',
              style: TextStyle(
                color: theme.colorScheme.primary,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // Connection status banner
          const RealtimeStatusBanner(),
          // Notification list
          Expanded(
            child: state.when(
              initial: () => const SizedBox.shrink(),
              loading: () => const Center(
                child: CircularProgressIndicator(),
              ),
              loaded: (notifications, unreadCount) {
                if (notifications.isEmpty) {
                  return _buildEmptyState(theme);
                }
                return RefreshIndicator(
                  onRefresh: () async {
                    await ref
                        .read(notificationProvider.notifier)
                        .loadNotifications();
                  },
                  child: ListView.separated(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: notifications.length,
                    separatorBuilder: (context, index) => Divider(
                      height: 1,
                      indent: 16,
                      endIndent: 16,
                      color: theme.colorScheme.outline.withValues(alpha: 0.1),
                    ),
                    itemBuilder: (context, index) {
                      final notification = notifications[index];
                      return NotificationTile(
                        notification: notification,
                        onMarkRead: () {
                          ref
                              .read(notificationProvider.notifier)
                              .markRead(notification.id);
                        },
                      );
                    },
                  ),
                );
              },
              error: (message) => _buildErrorState(theme, message),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState(ThemeData theme) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: theme.colorScheme.primary.withValues(alpha: 0.08),
              shape: BoxShape.circle,
            ),
            child: Icon(
              Icons.notifications_none_rounded,
              size: 40,
              color: theme.colorScheme.primary.withValues(alpha: 0.4),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'No notifications yet',
            style: theme.textTheme.titleMedium?.copyWith(
              color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'We\'ll notify you when there are stock\nupdates or deals nearby.',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurface.withValues(alpha: 0.4),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorState(ThemeData theme, String message) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.error_outline_rounded,
              size: 48,
              color: theme.colorScheme.error.withValues(alpha: 0.6),
            ),
            const SizedBox(height: 16),
            Text(
              'Something went wrong',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              message,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: () {
                ref.read(notificationProvider.notifier).loadNotifications();
              },
              icon: const Icon(Icons.refresh_rounded, size: 18),
              label: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }
}
