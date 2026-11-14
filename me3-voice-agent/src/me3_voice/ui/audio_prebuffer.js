/* Bounded startup jitter buffer for small PCM packets in the native browser path. */
(function (root) {
  class SwiftAudioPrebuffer {
    constructor(targetSeconds = 0.12, maxPackets = 16) {
      if (!Number.isFinite(targetSeconds) || targetSeconds < 0 || targetSeconds > 0.5) {
        throw new TypeError('invalid audio prebuffer target');
      }
      if (!Number.isInteger(maxPackets) || maxPackets < 1 || maxPackets > 20) {
        throw new TypeError('invalid audio prebuffer capacity');
      }
      this.targetSeconds = targetSeconds;
      this.maxPackets = maxPackets;
      this.queue = [];
      this.seconds = 0;
    }

    push(packet) {
      if (!packet || !Number.isFinite(packet.duration) || packet.duration <= 0) {
        throw new TypeError('invalid audio packet');
      }
      if (this.queue.length >= this.maxPackets) throw new RangeError('audio prebuffer full');
      this.queue.push(packet);
      this.seconds += packet.duration;
      return this.seconds >= this.targetSeconds ? this.take() : [];
    }

    take() {
      const packets = this.queue;
      this.queue = [];
      this.seconds = 0;
      return packets;
    }

    get length() { return this.queue.length; }

    clear() {
      this.queue = [];
      this.seconds = 0;
    }
  }

  root.SwiftAudioPrebuffer = SwiftAudioPrebuffer;
  if (typeof module !== 'undefined' && module.exports) module.exports = SwiftAudioPrebuffer;
})(typeof window !== 'undefined' ? window : globalThis);
