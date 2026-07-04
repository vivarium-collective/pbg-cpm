//! Local connectivity test for the CPM sweep. `count_components` is a pure
//! graph routine over a tiny node set (the same-owner neighbours of a flip
//! site); it decides whether removing the site would locally disconnect a cell.

pub fn count_components(members: &[usize], adjacent: &dyn Fn(usize, usize) -> bool) -> usize {
    let n = members.len();
    if n <= 1 {
        return n;
    }
    let mut seen = vec![false; n];
    let mut components = 0;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        components += 1;
        seen[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            for j in 0..n {
                if !seen[j] && adjacent(members[i], members[j]) {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
    }
    components
}

#[cfg(test)]
mod tests {
    use super::*;

    // adjacency from an explicit undirected edge list over node ids
    fn adj(edges: &'static [(usize, usize)]) -> impl Fn(usize, usize) -> bool {
        move |a, b| edges.iter().any(|&(u, v)| (u == a && v == b) || (u == b && v == a))
    }

    #[test]
    fn empty_and_single() {
        let none = adj(&[]);
        assert_eq!(count_components(&[], &none), 0);
        assert_eq!(count_components(&[7], &none), 1);
    }

    #[test]
    fn one_component_when_chained() {
        // 10-11-12 chain -> single component
        let a = adj(&[(10, 11), (11, 12)]);
        assert_eq!(count_components(&[10, 11, 12], &a), 1);
    }

    #[test]
    fn two_components_when_split() {
        // 10-11 and an isolated 12 -> two components (12 is only reachable via
        // the removed site, i.e. an articulation case)
        let a = adj(&[(10, 11)]);
        assert_eq!(count_components(&[10, 11, 12], &a), 2);
    }
}
