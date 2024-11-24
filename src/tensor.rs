use ndarray::{
    linalg::Dot, Array, ArrayBase, AsArray, Dimension, IxDyn, OwnedRepr, ShapeBuilder, ViewRepr,
};
use ndarray_rand::rand::Error;
use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
    hash::Hash,
    ops::{Add, Deref, Mul},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct IDCounter {
    id: usize,
}
static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Default, Debug, Clone)]
pub struct Tensor {
    pub data: Array<f32, IxDyn>,
    pub grad: Option<Array<f32, IxDyn>>,
    pub prev_op: Option<Operation>,
    pub id: usize,
}

impl Tensor
// where
//     A: Add<Output = A> + Clone, // A must support element-wise addition and cloning
//     A: Mul<Output = A> + Clone, // A must support element-wise multiplication and cloning
//     A: num_traits::identities::Zero,
// A: num_traits::identities::One,
// ArrayBase<OwnedRepr<A>, D>:
//     Dot<ArrayBase<OwnedRepr<A>, D>, Output = ArrayBase<OwnedRepr<A>, D>>,
{
    pub fn view(self) -> TensorView {
        TensorView {
            tensor: Rc::new(RefCell::new(self)),
        }
    }

    pub fn copy_clean(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
            grad: None,
            prev_op: None,
            id: self.id,
        }
    }
    pub fn new_with_data(data: Array<f32, IxDyn>) -> Tensor {
        Tensor {
            data: data,
            grad: None,
            prev_op: None,
            id: COUNTER.fetch_add(1, Ordering::SeqCst),
        }
    }
    pub fn new(
        data: Array<f32, IxDyn>,
        grad: Option<Array<f32, IxDyn>>,
        prev: Option<Operation>,
    ) -> Tensor {
        Tensor {
            data: data,
            grad: grad,
            prev_op: prev,
            id: COUNTER.fetch_add(1, Ordering::SeqCst),
        }
    }
    pub fn new_with_data_prev(data: Array<f32, IxDyn>, prev: Operation) -> Tensor {
        Tensor {
            data: data,
            grad: None,
            prev_op: Some(prev),
            id: COUNTER.fetch_add(1, Ordering::SeqCst),
        }
    }
    pub fn backward(&mut self, grad: Array<f32, IxDyn>) {
        self.grad = Some(
            self.grad
                .clone()
                .unwrap_or(Array::zeros(self.data.raw_dim()))
                + grad,
        );
        match self.prev_op.clone() {
            Some(Operation::Add(mut node)) => node.backward(self),
            Some(Operation::Mul(mut node)) => node.backward(self),
            None => {
                println!("No previous operation");
            }
        };
    }
    pub fn shape(&self) -> IxDyn {
        self.data.raw_dim()
    }
}

#[derive(Debug, Clone)]
pub enum Operation {
    Add(TensorAdd),
    Mul(TensorMul),
    // MatMul(TensorMatMul<A, D, E>),
}
fn inputs(operation: Operation) -> Vec<usize> {
    match operation {
        Operation::Add(node) => node.inputs(),
        Operation::Mul(node) => node.inputs(),
    }
}

pub struct TensorView {
    pub tensor: Rc<RefCell<Tensor>>,
}

// impl Deref for RefTensor<'a> {
//     type Target = Ref<Tensor>;
//     fn deref(&self) -> Self::Target {
//         self.tensor.borrow()
//     }
// }
// enum ComputationOperation<A, D> {
//     BinaryOp(Box<dyn BinaryOperation<A, D>>), // Node that performs a binary operation
//     UnaryOp(Box<dyn UnaryOperation<A, D>>),   // Node that performs a unary operation
//     TertiaryOp(Box<dyn TertiaryOperation<A, D>>),
// }

trait TertiaryOperation<D: Dimension> {
    fn forward(&self, input_a: Tensor, input_b: Tensor, input_c: Tensor) -> Tensor;
    fn backward(&self, input: Tensor) -> (Tensor, Tensor, Tensor);
}

trait UnaryOperation<A, D: Dimension> {
    fn forward(&self, input: Tensor) -> Tensor;
    fn backward(&self, input: Tensor) -> Tensor;
}

trait BinaryOperation<A, D: Dimension> {
    fn forward(input_a: Tensor, input_b: Tensor) -> Tensor;
    fn backward(&self, input: Tensor) -> (Tensor, Tensor);
}

// #[derive(Debug, Clone)]
// struct TensorMatMul<A, D: Dimension, E: Dimension> {
//     first: Rc<RefCell<Tensor< D>>>,
//     second: Rc<RefCell<Tensor< E>>>,
// }

// impl<A, D, E> TensorMatMul<A, D, E>
// where
//     A: Clone + num_traits::One, // A must support element-wise multiplication and cloning
//     D: Dimension,
//     E: Dimension,
//     ArrayBase<OwnedRepr<A>, D>:
//         Dot<ArrayBase<OwnedRepr<A>, E>, Output = ArrayBase<OwnedRepr<A>, E>>,
//     ArrayBase<OwnedRepr<A>, E>:
//         Dot<ArrayBase<OwnedRepr<A>, E>, Output = ArrayBase<OwnedRepr<A>, D>>,
// {
//     pub fn forward(input_a: Tensor< D>, input_b: Tensor< E>) -> Tensor< E> {
//         let data_a = input_a.data.clone();
//         let data_b = input_b.data.clone();

//         let data = data_a.dot(&data_b);

//         let node = TensorMatMul {
//             first: Rc::new(RefCell::new(input_a)),
//             second: Rc::new(RefCell::new(input_b)),
//         };
//         let result: Tensor< E> = Tensor {
//             data: data,
//             grad: None,
//             prev: Some(Operation::MatMul(node)),
//         };
//         result
//     }
//     pub fn backward(self, output: Tensor< E>) -> (Tensor< D>, Tensor< E>) {
//         let grad = output.grad.unwrap_or(Array::ones(output.data.raw_dim()));

//         let input_a = (*self.first).borrow().data.clone();
//         let input_b = (*self.second).borrow().data.clone();
//         let input_a_t = input_a.reversed_axes();

//         // Ckm = Sum_n Akn * Bnm
//         // dCkm/dBij = delta(m=j)* Aki
//         // dL/dB_ij = dL/dC_km*dC_km/dB_ij
//         // dL/dB_ij = dL/dC_kj*Aki
//         // dL/dB  = A^T @ (dL/dC)
//         let grad_b = input_a_t.dot(&grad);

//         // A @ B = C
//         // Ckm = Sum_n Akn * Bnm
//         // dCjk/dAlm = d_j==l * d_i==m  Bik
//         // dCjk/dAlm = Bmk

//         // dCkm/dAij = delta(k=i)*Bjm
//         // dL/dA_ij = dL/dC_km*dC_km/dA_ij
//         // dL/dA_ij = dL/dC_km*delta(k=i)*Bjm
//         // dL/dA_ij = dL/dC_im*Bjm
//         // dL/dA = (dL/dC) @ B
//         // dL/dA = B @ (dL/dC)^T
//         let grad_t = grad.reversed_axes();
//         let grad_a = input_b.dot(&grad_t);

//         (
//             Tensor {
//                 data: (*self.first).borrow().data.clone(),
//                 grad: Some(grad_a),
//                 prev: Some(Operation::MatMul(self.clone())),
//             },
//             Tensor {
//                 data: (*self.second).borrow().data.clone(),
//                 grad: Some(grad_b),
//                 prev: Some(Operation::MatMul(self.clone())),
//             },
//         )
//     }
// }

#[derive(Debug, Clone)]
struct TensorMul {
    first: Box<Tensor>,
    second: Box<Tensor>,
}

impl TensorMul
// where
// A: Mul<Output = A> + Clone + num_traits::One, // A must support element-wise multiplication and cloning
// D: Dimension,
// ArrayBase<OwnedRepr<A>, D>: Mul<Output = ArrayBase<OwnedRepr<A>, D>>, // Ensure element-wise multiplication is possible
// A: Add<Output = A> + Clone, // A must support element-wise addition and cloning
// A: Mul<Output = A> + Clone, // A must support element-wise multiplication and cloning
// A: num_traits::identities::Zero,
// A: num_traits::identities::One,
{
    pub fn forward(input_a: Tensor, input_b: Tensor) -> Tensor {
        let data = input_a.data.clone() * input_b.data.clone();
        let node = TensorMul {
            first: Box::new(input_a),
            second: Box::new(input_b),
        };
        Tensor::new_with_data_prev(data, Operation::Mul(node))
    }
    pub fn backward(&mut self, output: &mut Tensor) {
        let grad = output
            .grad
            .clone()
            .unwrap_or(Array::ones(output.data.raw_dim()));
        let grad_a = grad.clone() * self.second.data.clone();
        let grad_b = grad.clone() * self.first.data.clone();
        self.first.backward(grad_a);
        self.second.backward(grad_b);
    }
    pub fn inputs(self) -> Vec<usize> {
        vec![self.first.id, self.second.id]
    }
}

#[derive(Debug, Clone)]
struct TensorAdd {
    first: Box<Tensor>,
    second: Box<Tensor>,
}
impl TensorAdd {
    fn forward(input_a: Tensor, input_b: Tensor) -> Tensor {
        let data = input_a.data.clone() + input_b.data.clone();
        let node = TensorAdd {
            first: Box::new(input_a),
            second: Box::new(input_b),
        };
        Tensor::new_with_data_prev(data, Operation::Add(node))
    }
    fn backward(&mut self, output: &mut Tensor) {
        let grad = output
            .grad
            .clone()
            .unwrap_or(ndarray::Array::zeros(output.data.raw_dim()));
        let grad_a = grad.clone();
        let grad_b = grad.clone();
        self.first.backward(grad_a);
        self.second.backward(grad_b);
    }
    pub fn inputs(self) -> Vec<usize> {
        vec![self.first.id, self.second.id]
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        TensorAdd::forward(self, other)
    }
}

// impl<D: Dimension, A: Clone> Tensor< D>
// where
//     A: num_traits::One,
// {
//     pub fn mat_mul<E: Dimension>(self, other: Tensor< E>) -> Tensor< E>
//     where
//         ArrayBase<OwnedRepr<A>, D>:
//             Dot<ArrayBase<OwnedRepr<A>, E>, Output = ArrayBase<OwnedRepr<A>, E>>,
//         ArrayBase<OwnedRepr<A>, E>:
//             Dot<ArrayBase<OwnedRepr<A>, E>, Output = ArrayBase<OwnedRepr<A>, E>>,
//     {
//         TensorMatMul::<A, D, E>::forward(self, other)
//     }
// }

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        TensorMul::forward(self, other)
    }
}
pub fn add<A, D>(left: Array<A, D>, right: Array<A, D>) -> Array<A, D>
where
    A: Add<Output = A> + Clone,
    D: Dimension,
{
    left + right
}

// fn backward(mut loss_tensor: Tensor) {
//     let mut nodes = HashMap::new();
//     let mut depth: HashMap<usize, Vec<usize>> = HashMap::new();
//     let mut edges: HashMap<usize, Operation> = HashMap::new();
//     let mut next_layer: Vec<(Option<Operation>, usize)> = Vec::new();
//     let mut current_layer: Vec<(Option<Operation>, usize)> = Vec::new();
//     let loss_tensor_id = loss_tensor.id.clone();
//     current_layer.push((loss_tensor.prev_op.take(), loss_tensor_id));
//     nodes.insert(loss_tensor_id, loss_tensor);
//     while !current_layer.is_empty() {
//         for (operation, id) in current_layer {
//             if operation.is_none() {
//                 continue;
//             }
//             let operation = operation.unwrap();
//             let tensor = nodes.get_mut(&id).unwrap();
//             let inputs = inputs(operation);
//             for input in inputs {
//                 let input_tensor = nodes.get_mut(&input).unwrap();
//                 let input_depth = depth.get(&id).unwrap() + 1;
//                 let input_edges = edges.get_mut(&input).unwrap();
//                 input_edges.push(operation.clone());
//                 depth.insert(input, input_depth);
//                 next_layer.push((input_edges, input));
//             }
//         }
//         current_layer = next_layer;
//         next_layer = Vec::new();
//     }
// }
// fn construct(
//     tensor: &mut Tensor,
//     nodes: &mut HashMap<usize, Tensor>,
//     depth: &mut HashMap<usize, Vec<usize>>,
// ) -> Option<()> {
//     nodes.insert(tensor.id, tensor.copy_clean());

//     let operation = tensor.prev_op?;
//     let inputs = inputs(operation);
//     let depth = depth.get(&tensor.id)?;
//     for input in inputs {
//         let input_tensor = nodes.get(&input)?;
//         let input_depth = depth + 1;
//     }
//     None
// }

#[cfg(test)]
mod tests {
    use std::array;

    use ndarray::array;
    use ndarray::{linalg::Dot, Array, ArrayBase, AsArray, Dimension, OwnedRepr, ViewRepr};

    use super::*;

    #[test]
    fn test_complicated() {
        // let test_0: TensorView = Tensor::new_with_data(array![[1.0, 2.0, 3.0, 4.0]].into_dyn());
        // let test_1 = Tensor::new_with_data(array![[1.0, 2.0, 3.0, 4.0]].into_dyn());
        // let test_2 = test_0.clone() + test_1.clone();
        // let mut test_3: Tensor = test_2.clone() + test_1;
        // println!("forward: {:?}", test_2);
        // println!("_____________________________");
        // let start_grad = Array::ones(test_2.data.raw_dim());
        // test_3.backward(start_grad);

        // pretty_print(&test_2);
        // println!("{:?}", test_tensor2);
    }
    #[test]
    fn it_works() {
        let mut test = array![[1.0, 2.0, 3.0, 4.0]].into_dyn();
        let test2 = array![[1.0, 2.0, 3.0, 4.0]].into_dyn();
        let test3 = test.clone();

        test[[0, 0]] = 2.0;
        println!("{:?}", test);
        println!("{:?}", test3);
        panic!();
        let test_0 = Tensor::new_with_data(array![[1.0, 2.0, 3.0, 4.0]].into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = test_0.clone() + test_0; // grad_2 = 2 * test_1
        let mut test_2: Tensor = test_1.clone() * test_1;
        println!("forward: {:?}", test_2);
        println!("_____________________________");
        let start_grad = Array::ones(test_2.data.raw_dim());
        test_2.backward(start_grad);

        pretty_print(&test_2);
        // println!("{:?}", test_tensor2);
    }
    fn pretty_print(tensor: &Tensor) {
        println!(
            "Tensor: {:?}, Gradient {:?}, id: {:?}",
            tensor.data, tensor.grad, tensor.id
        );
        pretty_print_operation(tensor.prev_op.as_ref());
    }
    fn pretty_print_operation(operation: Option<&Operation>) {
        match operation {
            Some(Operation::Add(node)) => {
                println!("Addition");
                pretty_print(&node.first);
                pretty_print(&node.second);
            }
            Some(Operation::Mul(node)) => {
                println!("Multiplication");
                pretty_print(&node.first);
                pretty_print(&node.second);
            }
            None => {
                println!("None");
            }
        };
    }
}
