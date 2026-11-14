const assert = require('node:assert/strict');
const test = require('node:test');
const AudioPrebuffer = require('../src/me3_voice/ui/audio_prebuffer.js');

test('startup buffer releases ordered packets after 120 ms of PCM', () => {
  const buffer = new AudioPrebuffer(0.12, 16);
  const released = [];
  for (let id = 0; id < 5; id++) {
    released.push(...buffer.push({ id, duration: 0.02 }));
  }
  assert.deepEqual(released, []);
  released.push(...buffer.push({ id: 5, duration: 0.02 }));
  assert.deepEqual(released.map(packet => packet.id), [0, 1, 2, 3, 4, 5]);
  assert.equal(buffer.length, 0);
});

test('short final startup burst can be flushed by the bounded UI timer', () => {
  const buffer = new AudioPrebuffer(0.12, 16);
  buffer.push({ id: 1, duration: 0.02 });
  buffer.push({ id: 2, duration: 0.02 });
  assert.deepEqual(buffer.take().map(packet => packet.id), [1, 2]);
  assert.equal(buffer.length, 0);
});

test('buffer has a hard packet bound and clear discards interrupted audio', () => {
  const buffer = new AudioPrebuffer(0.5, 2);
  buffer.push({ id: 1, duration: 0.02 });
  buffer.push({ id: 2, duration: 0.02 });
  assert.throws(() => buffer.push({ id: 3, duration: 0.02 }), /full/);
  buffer.clear();
  assert.equal(buffer.length, 0);
  assert.deepEqual(buffer.take(), []);
});
