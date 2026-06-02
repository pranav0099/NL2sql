import 'package:freezed_annotation/freezed_annotation.dart';
import '../models/app_notification.dart';

part 'notification_state.freezed.dart';

@freezed
class NotificationState with _$NotificationState {
  const factory NotificationState.initial() = _Initial;
  const factory NotificationState.loading() = _Loading;
  const factory NotificationState.loaded({
    required List<AppNotification> notifications,
    @Default(0) int unreadCount,
  }) = _Loaded;
  const factory NotificationState.error(String message) = _Error;
}
