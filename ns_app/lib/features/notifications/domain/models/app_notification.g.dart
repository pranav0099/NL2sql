// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'app_notification.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$AppNotificationImpl _$$AppNotificationImplFromJson(
  Map<String, dynamic> json,
) => _$AppNotificationImpl(
  id: json['id'] as String,
  type: $enumDecode(_$NotificationTypeEnumMap, json['type']),
  title: json['title'] as String,
  body: json['body'] as String,
  retailerId: json['retailerId'] as String?,
  productId: json['productId'] as String?,
  sentAt: DateTime.parse(json['sentAt'] as String),
  isRead: json['isRead'] as bool? ?? false,
);

Map<String, dynamic> _$$AppNotificationImplToJson(
  _$AppNotificationImpl instance,
) => <String, dynamic>{
  'id': instance.id,
  'type': _$NotificationTypeEnumMap[instance.type]!,
  'title': instance.title,
  'body': instance.body,
  'retailerId': instance.retailerId,
  'productId': instance.productId,
  'sentAt': instance.sentAt.toIso8601String(),
  'isRead': instance.isRead,
};

const _$NotificationTypeEnumMap = {
  NotificationType.stockUpdate: 'stockUpdate',
  NotificationType.deal: 'deal',
};
