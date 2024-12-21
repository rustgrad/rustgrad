use ndarray::{Array, IxDyn};
use priority_queue::PriorityQueue;
use std::{
    cell::RefCell,
    cmp::Reverse,
    hash::{Hash, Hasher},
    ops::{Add, Deref, Mul},
    rc::Rc,
};

#[derive(Default, Debug, Clone)]
pub struct DataContainer {
    pub array: Array<f32, IxDyn>,
    pub grad: Option<Array<f32, IxDyn>>,

    pub num_consumers: usize,
}

#[derive(Default, Debug)]
pub struct Tensor {
    pub container: Rc<RefCell<DataContainer>>,
    prev_op: Option<Operation>,
}
pub struct HashTensor(Tensor);
// Separated so it is not possible to say tensor_a == tensor_b with the wrong implementation
impl From<Tensor> for HashTensor {
    fn from(value: Tensor) -> Self {
        HashTensor(value)
    }
}
impl From<HashTensor> for Tensor {
    fn from(value: HashTensor) -> Self {
        value.0
    }
}
impl Hash for HashTensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let container_adress = (*self.0.container).as_ptr() as usize;
        container_adress.hash(state);
    }
}
impl PartialEq for HashTensor {
    fn eq(&self, other: &Self) -> bool {
        let data_address = (*self.0.container).as_ptr() as usize;
        let other_data_address = (*other.0.container).as_ptr() as usize;
        data_address == other_data_address
    }
}
impl Eq for HashTensor {}

impl Tensor {
    pub fn new(data: Array<f32, IxDyn>) -> Tensor {
        let data = DataContainer {
            array: data,
            grad: None,
            num_consumers: 0,
        };
        Tensor {
            container: Rc::new(RefCell::new(data)),
            prev_op: None,
        }
    }
    fn new_with_prev(data: Array<f32, IxDyn>, prev_op: Operation) -> Tensor {
        let data = DataContainer {
            array: data,
            grad: None,
            num_consumers: 0,
        };
        Tensor {
            container: Rc::new(RefCell::new(data)),
            prev_op: Some(prev_op),
        }
    }
    pub fn backward(&mut self) {
        // This is a leaf node, we need to build the graph
        self.build_graph();

        let start_grad = Array::ones(self.shape());
        self.backward_internal(start_grad);
    }

    pub fn queue_backward(&mut self) {
        self.build_graph();

        let start_grad = Array::ones(self.shape());
        self.queue_backward_internal(start_grad);
    }

    pub fn queue_backward_internal(&self, grad: Array<f32, IxDyn>) {
        self.add_grad(grad);
        let mut queue: PriorityQueue<HashTensor, Reverse<usize>> = PriorityQueue::new();
        queue.push(self.clone().into(), Reverse(self.consumers()));
        while !queue.is_empty() {
            let (hash_tensor, priority) = queue.pop().expect("Queue should not be empty");
            let tensor: Tensor = hash_tensor.into();
            if priority.0 > 0 {
                panic!("Did not find ready tensor");
            }
            let operation = tensor.prev_op.clone();
            if operation.is_none() {
                continue;
            }
            let grads = operation.unwrap().grad(&tensor);
            for (grad, tensor) in grads {
                tensor.add_grad(grad);
                let priority = Reverse(tensor.consumers());
                queue.push(tensor.into(), priority);
            }
        }
    }

    pub fn backward_internal(&self, grad: Array<f32, IxDyn>) {
        self.add_grad(grad);
        if self.container.borrow().num_consumers > 0 {
            return;
        }
        match self.prev_op.clone() {
            Some(Operation::Add(node)) => node.backward(self),
            Some(Operation::Mul(node)) => node.backward(self),
            None => {
                println!("No previous operation");
            }
        };
    }
    pub fn zero_graph(&mut self) {
        self.container.deref().borrow_mut().num_consumers = 0;
        match self.prev_op.clone() {
            Some(Operation::Add(mut node)) => {
                node.first.zero_graph();
                node.second.zero_graph();
            }
            Some(Operation::Mul(mut node)) => {
                node.first.zero_graph();
                node.second.zero_graph();
            }
            None => {
                println!("No previous operation");
            }
        };
    }

    pub fn build_graph(&self) {
        self.container.deref().borrow_mut().num_consumers += 1;
        if self.container.borrow().num_consumers > 1 {
            return;
        } else {
            match self.prev_op.clone() {
                Some(Operation::Add(mut node)) => {
                    node.first.build_graph();
                    node.second.build_graph();
                }
                Some(Operation::Mul(mut node)) => {
                    node.first.build_graph();
                    node.second.build_graph();
                }
                None => {
                    println!("No previous operation");
                }
            };
        }
    }

    fn add_grad(&self, grad: Array<f32, IxDyn>) {
        let new_grad = self.grad().unwrap_or(Array::zeros(self.shape())).clone() + grad.clone();
        self.container.deref().borrow_mut().grad = Some(new_grad);
        self.container.deref().borrow_mut().num_consumers -= 1;
    }
    fn consumers(&self) -> usize {
        (*self.container).borrow().num_consumers
    }

    pub fn shape(&self) -> IxDyn {
        self.container.borrow().array.raw_dim()
    }
    pub fn grad(&self) -> Option<Array<f32, IxDyn>> {
        self.container.borrow().grad.clone()
    }
    pub fn data(&self) -> Array<f32, IxDyn> {
        self.container.borrow().array.clone()
    }
}
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            container: self.container.clone(),
            prev_op: self.prev_op.clone(),
        }
    }
}

#[derive(Debug, Clone)]
enum Operation {
    Add(TensorAdd),
    Mul(TensorMul),
    // MatMul(TensorMatMul<A, D, E>),
}
impl Grad for Operation {
    fn grad(&self, output: &Tensor) -> Vec<(Array<f32, IxDyn>, Tensor)> {
        match self {
            Operation::Add(tensor_add) => tensor_add.grad(output),
            Operation::Mul(tensor_mul) => tensor_mul.grad(output),
        }
    }
}

trait Grad {
    fn grad(&self, output: &Tensor) -> Vec<(Array<f32, IxDyn>, Tensor)>;
}

#[derive(Debug, Clone)]
struct TensorMul {
    first: Box<Tensor>,
    second: Box<Tensor>,
}

impl TensorMul {
    pub fn forward(input_a: Tensor, input_b: Tensor) -> Tensor {
        let result =
            input_a.container.borrow().array.clone() * input_b.container.borrow().array.clone();
        let node = TensorMul {
            first: Box::new(input_a),
            second: Box::new(input_b),
        };
        Tensor::new_with_prev(result, Operation::Mul(node))
    }
    pub fn grad(&self, output: &Tensor) -> Vec<(Array<f32, IxDyn>, Tensor)> {
        let grad = output.container.borrow().grad.clone();
        let grad = grad.unwrap_or(Array::ones(output.shape()));
        [self.first.clone(), self.second.clone()]
            .iter()
            .map(|tensor| {
                (
                    tensor.container.borrow().array.clone() * grad.clone(),
                    (**tensor).clone(),
                )
            })
            .collect()
    }
    pub fn backward(&self, output: &Tensor) {
        for (grad, tensor) in self.grad(output) {
            tensor.backward_internal(grad);
        }
    }
}

#[derive(Debug, Clone)]
struct TensorAdd {
    first: Box<Tensor>,
    second: Box<Tensor>,
}
impl TensorAdd {
    fn forward(input_a: Tensor, input_b: Tensor) -> Tensor {
        let result =
            input_a.container.borrow().array.clone() + input_b.container.borrow().array.clone();
        let node = TensorAdd {
            first: Box::new(input_a),
            second: Box::new(input_b),
        };
        Tensor::new_with_prev(result, Operation::Add(node))
    }
    fn grad(&self, output: &Tensor) -> Vec<(Array<f32, IxDyn>, Tensor)> {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::zeros(output.shape()));
        [self.first.clone(), self.second.clone()]
            .iter()
            .map(|tensor| (grad.clone(), (**tensor).clone()))
            .collect()
    }
    fn backward(&self, output: &Tensor) {
        for (grad, tensor) in self.grad(output) {
            tensor.backward_internal(grad);
        }
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        TensorAdd::forward(self, other)
    }
}
impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        TensorMul::forward(self, other)
    }
}
#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn it_works_2() {
        let test_0 = Tensor::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = test_0.clone() + test_0.clone(); // grad_2 = 2 * test_1
        let mut test_2: Tensor = test_1.clone() * test_1.clone();
        println!("forward: {:?}", test_2);
        println!("_____________________________");
        test_2.queue_backward();

        assert_eq!(
            test_0.grad(),
            Some(array![[8.0, 16.0, 24.0, 32.0]].into_dyn())
        );
        assert_eq!(
            test_1.grad(),
            Some(array![[4.0, 8.0, 12.0, 16.0]].into_dyn())
        );
        assert_eq!(test_2.grad(), Some(array![[1.0, 1.0, 1.0, 1.0]].into_dyn()));

        println!("{:?}", test_2);
    }

    #[test]
    fn it_works() {
        let test_0 = Tensor::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = test_0.clone() + test_0.clone(); // grad_2 = 2 * test_1
        let mut test_2: Tensor = test_1.clone() * test_1.clone();
        println!("forward: {:?}", test_2);
        println!("_____________________________");
        test_2.backward();

        assert_eq!(
            test_0.grad(),
            Some(array![[8.0, 16.0, 24.0, 32.0]].into_dyn())
        );
        assert_eq!(
            test_1.grad(),
            Some(array![[4.0, 8.0, 12.0, 16.0]].into_dyn())
        );
        assert_eq!(test_2.grad(), Some(array![[1.0, 1.0, 1.0, 1.0]].into_dyn()));

        println!("{:?}", test_2);
    }
}
