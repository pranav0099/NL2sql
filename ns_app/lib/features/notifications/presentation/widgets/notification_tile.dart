import 'package:flutter/material.dart';
import '../../domain/models/app_notification.dart';
import '../../../../core/notifications/notification_router.dart';

/// A single notification row in the notification inbox.
///
/// - Leading: coloured icon based on notification type
/// - Blue left border when unread
/// - Bold title when unread
/// - Relative time trailing text
/// - Swipe to dismiss → marks as read
class NotificationTile extends StatelessWidget {
  final AppNotification notification;
  final VoidCallback? onMarkRead;
  final VoidCallback? onTap;

  const NotificationTile({
    super.key,
    required this.notification,
    this.onMarkRead,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isUnread = !notification.isRead;

    return Dismissible(
      key: ValueKey(notification.id),
      direction: DismissDirection.endToStart,
      background: Container(
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 20),
        color: theme.colorScheme.primary.withValues(alpha: 0.1),
        child: Icon(
          Icons.done_all_rounded,
          color: theme.colorScheme.primary,
        ),
      ),
      onDismissed: (_) => onMarkRead?.call(),
      child: Container(
        decoration: BoxDecoration(
          border: Border(
            left: BorderSide(
              color: isUnread
                  ? theme.colorScheme.primary
                  : Colors.transparent,
              width: 3,
            ),
          ),
        ),
        child: ListTile(
          onTap: onTap ?? () {
            NotificationRouter.route({
              'type': notification.type == NotificationType.stockUpdate
                  ? 'STOCK_UPDATE'
                  : 'DEAL',
              if (notification.retailerId != null)
                'retailerId': notification.retailerId!,
              if (notification.productId != null)
                'productId': notification.productId!,
            });
          },
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          leading: _buildLeadingIcon(theme),
          title: Text(
            notification.title,
            style: theme.textTheme.bodyLarge?.copyWith(
              fontWeight: isUnread ? FontWeight.w700 : FontWeight.w400,
              color: theme.colorScheme.onSurface,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          subtitle: Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              notification.body,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          trailing: Text(
            _relativeTime(notification.sentAt),
            style: theme.textTheme.labelSmall?.copyWith(
              color: theme.colorScheme.onSurface.withValues(alpha: 0.45),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLeadingIcon(ThemeData theme) {
    final IconData icon;
    final Color color;
    final Color bgColor;

    switch (notification.type) {
      case NotificationType.stockUpdate:
        icon = Icons.inventory_2_rounded;
        // Blue for back-in-stock style, red for out-of-stock.
        final isPositive = notification.title.toLowerCase().contains('back') ||
            notification.title.toLowerCase().contains('available');
        color = isPositive
            ? const Color(0xFF2563EB)
            : const Color(0xFFEF4444);
        bgColor = color.withValues(alpha: 0.1);
        break;
      case NotificationType.deal:
        icon = Icons.local_offer_rounded;
        color = const Color(0xFFF59E0B);
        bgColor = color.withValues(alpha: 0.1);
        break;
    }

    return Container(
      width: 44,
      height: 44,
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Icon(icon, color: color, size: 22),
    );
  }

  String _relativeTime(DateTime sentAt) {
    final diff = DateTime.now().difference(sentAt);

    if (diff.inSeconds < 60) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${(diff.inDays / 7).floor()}w ago';
  }
}
