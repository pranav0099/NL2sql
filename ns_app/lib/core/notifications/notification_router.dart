import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

/// Routes the user to the correct screen when a notification is tapped.
///
/// Works in all three app states: foreground, background, and terminated.
/// Requires [navigatorKey] to be attached to the root MaterialApp.
class NotificationRouter {
  static final GlobalKey<NavigatorState> navigatorKey =
      GlobalKey<NavigatorState>();

  /// Route from a parsed data map (FCM / STOMP payload).
  static void route(Map<String, dynamic> data) {
    final type = data['type'] as String? ?? '';
    final retailerId = data['retailerId'] as String?;

    final context = navigatorKey.currentContext;
    if (context == null) {
      debugPrint('[NotificationRouter] No navigator context — skipping route.');
      return;
    }

    switch (type) {
      case 'STOCK_UPDATE':
      case 'DEAL':
        if (retailerId != null) {
          GoRouter.of(context).push('/customer/retailer/$retailerId');
        } else {
          GoRouter.of(context).push('/customer/notifications');
        }
        break;
      default:
        GoRouter.of(context).push('/customer/notifications');
    }
  }

  /// Route from a raw JSON string payload (flutter_local_notifications tap).
  static void handlePayload(String payload) {
    try {
      final data = jsonDecode(payload) as Map<String, dynamic>;
      route(data);
    } catch (e) {
      debugPrint('[NotificationRouter] Failed to parse payload: $e');
      // Fall back to notifications screen
      final context = navigatorKey.currentContext;
      if (context != null) {
        GoRouter.of(context).push('/customer/notifications');
      }
    }
  }
}
