use rtrb::{Consumer, Producer, RingBuffer};

/// A utility struct to bundle the producer and consumer interaction
/// functionality if we want to abstract over the specific library.
/// For now, we mainly expose the rtrb types or a constructor.

pub type AudioProducer = Producer<f32>;
pub type AudioConsumer = Consumer<f32>;

/// Creates a new pair of (Producer, Consumer) with the specified capacity.
pub fn make_audio_ring_buffer(capacity: usize) -> (AudioProducer, AudioConsumer) {
    RingBuffer::new(capacity)
}
