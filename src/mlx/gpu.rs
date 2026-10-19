//! Compatibility entry points for callers that previously created a CUDA stream.
use anyhow::Result;
pub struct Stream;
impl Stream {
    pub fn new() -> Result<Self> {
        crate::apple::init()?;
        Ok(Self)
    }
    pub fn sync(&self) -> Result<()> {
        Ok(())
    }
}
pub fn mem_info() -> Result<(usize, usize)> {
    crate::apple::memory_info()
}
