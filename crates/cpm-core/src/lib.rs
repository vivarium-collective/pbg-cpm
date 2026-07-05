pub type CellId = u32;
pub const MEDIUM: CellId = 0;

pub mod lattice;
pub mod world;
pub mod energy;
pub mod field;
pub mod sweep;
pub mod mitosis;
pub mod init;
pub mod connectivity;
pub mod membrane;
pub mod junction;
pub mod length;
pub mod external;
pub mod parallel;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn medium_is_zero() {
        assert_eq!(MEDIUM, 0 as CellId);
    }
}
