use std::sync::mpsc;

pub trait ParamEventAdapter {
    fn on_param_value_changed(&self, id: &str);
    fn on_param_modulation_changed(&self, id: &str, modulation_offset: f32);
    fn on_param_values_changed(&self);
}

#[derive(Clone)]
pub struct ParamEventDispatcher<Message>
where
    Message: Send + 'static,
{
    tx: Option<mpsc::Sender<Message>>,
}

impl<Message> Default for ParamEventDispatcher<Message>
where
    Message: Send + 'static,
{
    fn default() -> Self {
        Self { tx: None }
    }
}

impl<Message> ParamEventDispatcher<Message>
where
    Message: Send + 'static,
{
    pub fn set_sender(&mut self, tx: mpsc::Sender<Message>) {
        self.tx = Some(tx);
    }

    pub fn sender(&self) -> Option<mpsc::Sender<Message>> {
        self.tx.clone()
    }

    pub fn send_with<F>(&self, map: F)
    where
        F: FnOnce() -> Message,
    {
        if let Some(tx) = &self.tx {
            let _ = tx.send(map());
        }
    }
}
