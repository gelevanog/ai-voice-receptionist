// Microphone -> 16 kHz mono PCM16, posted in 20 ms frames (320 samples) to the main thread.
// Downsampling averages the input over each output period (a box low-pass), enough for speech recognition.
class Mic16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0;
    this.count = 0;
    this.pos = 0;
    this.frame = new Int16Array(320);
    this.filled = 0;
    this.levelSum = 0;
    this.levelN = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    for (let i = 0; i < channel.length; i++) {
      const s = channel[i];
      this.acc += s;
      this.count += 1;
      this.pos += 1;
      this.levelSum += s * s;
      this.levelN += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.count));
        this.acc = 0;
        this.count = 0;
        this.frame[this.filled++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.filled === this.frame.length) {
          const out = this.frame.slice(0);
          this.port.postMessage({ pcm: out.buffer, level: Math.sqrt(this.levelSum / this.levelN) }, [out.buffer]);
          this.filled = 0;
          this.levelSum = 0;
          this.levelN = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("mic-16k", Mic16k);
