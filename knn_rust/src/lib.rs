#[cfg(feature = "python-module")]
extern crate pyo3;

use std::collections::HashMap;

#[cfg(feature = "python-module")]
use pyo3::prelude::*;

const MIN_FLOAT_DIFFERENCE: f64 = 1E-10;

pub enum WeightMethod {
    Majority,
    Weighted,
}

pub enum DistanceMethod {
    Manhattan,
    Euclidean,
}

#[cfg(feature = "python-module")]
#[pyclass]
pub struct KNearestNeighbors {
    pub weight_method: WeightMethod,
    pub n_neighbors: usize,
    pub distance_method: DistanceMethod,
    class_vector: Vec<i64>,
    data: Vec<Vec<f64>>,
}

#[cfg(not(feature = "python-module"))]
pub struct KNearestNeighbors {
    pub weight_method: WeightMethod,
    pub n_neighbors: usize,
    pub distance_method: DistanceMethod,
    class_vector: Vec<i64>,
    data: Vec<Vec<f64>>,
}


impl KNearestNeighbors {
    #[cfg(not(feature = "python-module"))]
    #[inline]
    pub fn classify(&self, query: &[f64]) -> i64 {
        self.classify_(query)
    }

    pub fn classify_(&self, query: &[f64]) -> i64 {
        let neighbors_with_classes: Vec<_> = self
            .nearest_neighbors(&query)
            .iter()
            .map(|(index, distance)| (self.class_vector[*index], *index, *distance))
            .collect();

        if neighbors_with_classes.len() == 1 {
            return neighbors_with_classes[0].0;
        }

        match self.weight_method {
            WeightMethod::Majority => {
                // Create map to store the amount of occurences of the classes
                let mut classes_count: HashMap<i64, u64> = HashMap::new();
                neighbors_with_classes.iter().for_each(|(class, _, _)| {
                    // Add a class to the map with count 1 if it is not in the map, otherwise increment the count
                    match classes_count.get_mut(class) {
                        None => {
                            classes_count.insert(*class, 1);
                        }
                        Some(count) => {
                            *count += 1;
                        }
                    }
                });

                // Return the class with the highest count
                classes_count
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .unwrap()
                    .0
            }

            WeightMethod::Weighted => {
                // Determine the closest and the furthest neighbor
                let closest_neighbor = neighbors_with_classes
                    .iter()
                    .min_by(|(_, _, d0), (_, _, d1)| d0.partial_cmp(d1).unwrap())
                    .unwrap();
                let furthest_neighbor = neighbors_with_classes
                    .iter()
                    .max_by(|(_, _, d0), (_, _, d1)| d0.partial_cmp(d1).unwrap())
                    .unwrap();
                let mut classes_weights: HashMap<i64, f64> = HashMap::new();

                neighbors_with_classes
                    .iter()
                    .for_each(|(class, _index, d_query_current)| {
                        let d_query_furthest = furthest_neighbor.2;
                        let d_query_closest = closest_neighbor.2;
                        // calculate the new weight for the class
                        let new_weight =
                            if (d_query_furthest - d_query_closest).abs() < MIN_FLOAT_DIFFERENCE {
                                1f64
                            } else {
                                (d_query_furthest - *d_query_current)
                                    / (d_query_furthest - d_query_closest)
                            };

                        // Store it the same way as with the majority rule
                        match classes_weights.get_mut(class) {
                            None => {
                                classes_weights.insert(*class, new_weight);
                            }
                            Some(weight) => {
                                *weight += new_weight;
                            }
                        }
                    });
                // Return the class with the highest sum of weights
                classes_weights
                    .into_iter()
                    .max_by(|(_, weight0), (_, weight1)| weight0.partial_cmp(weight1).unwrap())
                    .unwrap()
                    .0
            }
        }
    }

    pub fn new(class_vector: Vec<i64>, data: Vec<Vec<f64>>) -> Self {
        KNearestNeighbors {
            weight_method: WeightMethod::Majority,
            n_neighbors: 5,
            distance_method: DistanceMethod::Euclidean,
            class_vector,
            data,
        }
    }

    fn distance(&self, p0: &[f64], p1: &[f64]) -> f64 {
        assert_eq!(p0.len(), p1.len());
        let pow_i = match &self.distance_method {
            DistanceMethod::Manhattan => 1,
            DistanceMethod::Euclidean => 2,
        };
        p0.iter()
            .zip(p1)
            .map(|(v0, v1)| (*v0 - *v1).abs().powi(pow_i))
            .sum()
    }

    pub fn with_distance_method(mut self, value: DistanceMethod) -> Self {
        self.distance_method = value;
        self
    }

    pub fn with_weight_method(mut self, value: WeightMethod) -> Self {
        self.weight_method = value;
        self
    }

    pub fn with_n_neighbors(mut self, value: usize) -> Self {
        self.n_neighbors = value;
        self
    }


    fn nearest_neighbors(&self, query: &[f64]) -> Vec<(usize, f64)> {
        let mut dist_with_indexes: Vec<(usize, f64)> = self
            .data
            .iter()
            .map(|data_point| {
                assert_eq!(data_point.len(), query.len());
                self.distance(query, data_point)
            })
            .enumerate()
            .collect();
        let mut min_dist_neighbors: Vec<(usize, f64)> =
            Vec::with_capacity(self.n_neighbors);
        assert!(self.n_neighbors <= self.data.len());
        // if n_neighbors > log2(data_points), sorting is used, otherwise the maximum value is picked n times
        if (self.n_neighbors as f32) < (dist_with_indexes.len() as f32).log2() {
            let mut added_indexes: Vec<usize> = Vec::with_capacity(self.n_neighbors);
            while min_dist_neighbors.len() < self.n_neighbors {
                min_dist_neighbors.push(
                    *dist_with_indexes
                        .iter()
                        .enumerate()
                        // Filter out the indexes which are already added
                        .filter(|(index, _)| {
                            !added_indexes.contains(index)
                            // Discard the indexes
                        }).map(|(_, item)| {
                        item
                    })
                        // Get the item with the minimal distance
                        .min_by(|(_, v0), (_, v1)|
                            v0.partial_cmp(v1).unwrap())
                        .unwrap()
                );
                added_indexes.push(min_dist_neighbors.last().unwrap().0)
            }
        } else {
            dist_with_indexes.sort_unstable_by(|(_, v0), (_, v1)| v0.partial_cmp(v1).unwrap());
            for min_dist_neighbor in dist_with_indexes
                .into_iter()
                .take(self.n_neighbors) {
                min_dist_neighbors.push(min_dist_neighbor);
            }
        }
        min_dist_neighbors
    }
}


#[cfg(feature = "python-module")]
#[pymethods]
impl KNearestNeighbors {
    #[new]
    fn new_py(obj: &PyRawObject, data: Vec<Vec<f64>>, class_vector: Vec<i64>) {
        obj.init({
            Self::new(class_vector, data)
        });
    }

    fn set_weight_method(&mut self, weight_method: &str) {
        match weight_method.to_lowercase().as_ref() {
            "weighted" => self.weight_method = WeightMethod::Weighted,
            "majority" => self.weight_method = WeightMethod::Majority,
            _ => panic!("invalid weight method, weight method should be majority or weighted")
        }
    }

    fn set_distance_method(&mut self, distance_method: &str) {
        match distance_method.to_lowercase().as_ref() {
            "euclidean" => self.distance_method = DistanceMethod::Euclidean,
            "manhattan" => self.distance_method = DistanceMethod::Manhattan,
            _ => panic!("invalid distance method, weight method should be euclidean or manhattan")
        }
    }


    fn set_n_neighbors(&mut self, n_neighbors: usize) {
        self.n_neighbors = n_neighbors;
    }

    fn classify(&self, query: Vec<f64>) -> i64 {
        self.classify_(&query)
    }
}

#[cfg(feature = "python-module")]
#[pymodule]
fn knn_rust(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<KNearestNeighbors>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let arr = vec![
            vec![2.0, 3.0, 4.0],
            vec![12.0, 14.0, 13.0],
            vec![15.324_234, 16.2, 17.9]
        ];
        let mut classifier = KNearestNeighbors::new(vec![1, 1, 2], arr)
            .with_n_neighbors(2)
            .with_weight_method(WeightMethod::Majority);
        let result0 = classifier.classify_(&[5.0, 6.0, 7.0]);
        assert_eq!(result0, 1);
        classifier.weight_method = WeightMethod::Weighted;
        classifier.n_neighbors = 3;
        let result1 = classifier.classify(&[21.0, 18.0, 18.0]);
        assert_eq!(result1, 2);
    }

}
