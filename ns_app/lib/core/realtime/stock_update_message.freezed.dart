// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'stock_update_message.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

StockUpdateMessage _$StockUpdateMessageFromJson(Map<String, dynamic> json) {
  return _StockUpdateMessage.fromJson(json);
}

/// @nodoc
mixin _$StockUpdateMessage {
  String get productId => throw _privateConstructorUsedError;
  String get productName => throw _privateConstructorUsedError;
  String get retailerId => throw _privateConstructorUsedError;
  StockLevel get stockLevel => throw _privateConstructorUsedError;
  DateTime get updatedAt => throw _privateConstructorUsedError;

  /// Serializes this StockUpdateMessage to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of StockUpdateMessage
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $StockUpdateMessageCopyWith<StockUpdateMessage> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $StockUpdateMessageCopyWith<$Res> {
  factory $StockUpdateMessageCopyWith(
    StockUpdateMessage value,
    $Res Function(StockUpdateMessage) then,
  ) = _$StockUpdateMessageCopyWithImpl<$Res, StockUpdateMessage>;
  @useResult
  $Res call({
    String productId,
    String productName,
    String retailerId,
    StockLevel stockLevel,
    DateTime updatedAt,
  });
}

/// @nodoc
class _$StockUpdateMessageCopyWithImpl<$Res, $Val extends StockUpdateMessage>
    implements $StockUpdateMessageCopyWith<$Res> {
  _$StockUpdateMessageCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of StockUpdateMessage
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? productId = null,
    Object? productName = null,
    Object? retailerId = null,
    Object? stockLevel = null,
    Object? updatedAt = null,
  }) {
    return _then(
      _value.copyWith(
            productId: null == productId
                ? _value.productId
                : productId // ignore: cast_nullable_to_non_nullable
                      as String,
            productName: null == productName
                ? _value.productName
                : productName // ignore: cast_nullable_to_non_nullable
                      as String,
            retailerId: null == retailerId
                ? _value.retailerId
                : retailerId // ignore: cast_nullable_to_non_nullable
                      as String,
            stockLevel: null == stockLevel
                ? _value.stockLevel
                : stockLevel // ignore: cast_nullable_to_non_nullable
                      as StockLevel,
            updatedAt: null == updatedAt
                ? _value.updatedAt
                : updatedAt // ignore: cast_nullable_to_non_nullable
                      as DateTime,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$StockUpdateMessageImplCopyWith<$Res>
    implements $StockUpdateMessageCopyWith<$Res> {
  factory _$$StockUpdateMessageImplCopyWith(
    _$StockUpdateMessageImpl value,
    $Res Function(_$StockUpdateMessageImpl) then,
  ) = __$$StockUpdateMessageImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    String productId,
    String productName,
    String retailerId,
    StockLevel stockLevel,
    DateTime updatedAt,
  });
}

/// @nodoc
class __$$StockUpdateMessageImplCopyWithImpl<$Res>
    extends _$StockUpdateMessageCopyWithImpl<$Res, _$StockUpdateMessageImpl>
    implements _$$StockUpdateMessageImplCopyWith<$Res> {
  __$$StockUpdateMessageImplCopyWithImpl(
    _$StockUpdateMessageImpl _value,
    $Res Function(_$StockUpdateMessageImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of StockUpdateMessage
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? productId = null,
    Object? productName = null,
    Object? retailerId = null,
    Object? stockLevel = null,
    Object? updatedAt = null,
  }) {
    return _then(
      _$StockUpdateMessageImpl(
        productId: null == productId
            ? _value.productId
            : productId // ignore: cast_nullable_to_non_nullable
                  as String,
        productName: null == productName
            ? _value.productName
            : productName // ignore: cast_nullable_to_non_nullable
                  as String,
        retailerId: null == retailerId
            ? _value.retailerId
            : retailerId // ignore: cast_nullable_to_non_nullable
                  as String,
        stockLevel: null == stockLevel
            ? _value.stockLevel
            : stockLevel // ignore: cast_nullable_to_non_nullable
                  as StockLevel,
        updatedAt: null == updatedAt
            ? _value.updatedAt
            : updatedAt // ignore: cast_nullable_to_non_nullable
                  as DateTime,
      ),
    );
  }
}

/// @nodoc
@JsonSerializable()
class _$StockUpdateMessageImpl implements _StockUpdateMessage {
  const _$StockUpdateMessageImpl({
    required this.productId,
    required this.productName,
    required this.retailerId,
    required this.stockLevel,
    required this.updatedAt,
  });

  factory _$StockUpdateMessageImpl.fromJson(Map<String, dynamic> json) =>
      _$$StockUpdateMessageImplFromJson(json);

  @override
  final String productId;
  @override
  final String productName;
  @override
  final String retailerId;
  @override
  final StockLevel stockLevel;
  @override
  final DateTime updatedAt;

  @override
  String toString() {
    return 'StockUpdateMessage(productId: $productId, productName: $productName, retailerId: $retailerId, stockLevel: $stockLevel, updatedAt: $updatedAt)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$StockUpdateMessageImpl &&
            (identical(other.productId, productId) ||
                other.productId == productId) &&
            (identical(other.productName, productName) ||
                other.productName == productName) &&
            (identical(other.retailerId, retailerId) ||
                other.retailerId == retailerId) &&
            (identical(other.stockLevel, stockLevel) ||
                other.stockLevel == stockLevel) &&
            (identical(other.updatedAt, updatedAt) ||
                other.updatedAt == updatedAt));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(
    runtimeType,
    productId,
    productName,
    retailerId,
    stockLevel,
    updatedAt,
  );

  /// Create a copy of StockUpdateMessage
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$StockUpdateMessageImplCopyWith<_$StockUpdateMessageImpl> get copyWith =>
      __$$StockUpdateMessageImplCopyWithImpl<_$StockUpdateMessageImpl>(
        this,
        _$identity,
      );

  @override
  Map<String, dynamic> toJson() {
    return _$$StockUpdateMessageImplToJson(this);
  }
}

abstract class _StockUpdateMessage implements StockUpdateMessage {
  const factory _StockUpdateMessage({
    required final String productId,
    required final String productName,
    required final String retailerId,
    required final StockLevel stockLevel,
    required final DateTime updatedAt,
  }) = _$StockUpdateMessageImpl;

  factory _StockUpdateMessage.fromJson(Map<String, dynamic> json) =
      _$StockUpdateMessageImpl.fromJson;

  @override
  String get productId;
  @override
  String get productName;
  @override
  String get retailerId;
  @override
  StockLevel get stockLevel;
  @override
  DateTime get updatedAt;

  /// Create a copy of StockUpdateMessage
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$StockUpdateMessageImplCopyWith<_$StockUpdateMessageImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
