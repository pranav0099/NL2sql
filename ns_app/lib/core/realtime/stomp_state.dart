import 'package:freezed_annotation/freezed_annotation.dart';

part 'stomp_state.freezed.dart';

@freezed
class StompState with _$StompState {
  const factory StompState.disconnected() = _Disconnected;
  const factory StompState.connecting() = _Connecting;
  const factory StompState.connected(String sessionId) = _Connected;
  const factory StompState.error(String message) = _Error;
}
