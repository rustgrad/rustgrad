use ndarray::{Array, IxDyn};
use std::{
    cell::RefCell,
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
        // This is a leaf node, we need to build the graphs
        let start_grad = Array::ones(self.shape());
        self.add_consumer();
        self.backward_internal(start_grad);
    }

    pub fn backward_internal(&mut self, grad: Array<f32, IxDyn>) {
        let new_grad = self.grad().unwrap_or(Array::zeros(self.shape())).clone() + grad.clone();
        self.container.deref().borrow_mut().grad = Some(new_grad);
        self.remove_consumer();
        if self.container.borrow().num_consumers > 0 {
            return;
        }
        match self.prev_op.clone() {
            Some(Operation::Add(mut node)) => node.backward(self),
            Some(Operation::Mul(mut node)) => node.backward(self),
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

    pub fn add_consumer(&mut self) {
        self.container.deref().borrow_mut().num_consumers += 1;
    }
    pub fn remove_consumer(&mut self) {
        self.container.deref().borrow_mut().num_consumers -= 1;
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

#[derive(Debug, Clone)]
struct TensorMul {
    first: Box<Tensor>,
    second: Box<Tensor>,
}

impl TensorMul {
    pub fn forward(mut input_a: Tensor, mut input_b: Tensor) -> Tensor {
        input_a.add_consumer();
        input_b.add_consumer();
        let result =
            input_a.container.borrow().array.clone() * input_b.container.borrow().array.clone();
        let node = TensorMul {
            first: Box::new(input_a),
            second: Box::new(input_b),
        };
        Tensor::new_with_prev(result, Operation::Mul(node))
    }
    pub fn backward(&mut self, output: &mut Tensor) {
        let grad = output.container.borrow().grad.clone();
        let grad = grad.unwrap_or(Array::ones(output.shape()));
        let grad_a = grad.clone() * self.second.container.borrow().array.clone();
        let grad_b = grad.clone() * self.first.container.borrow().array.clone();
        self.first.backward_internal(grad_a);
        self.second.backward_internal(grad_b);
    }
}

#[derive(Debug, Clone)]
struct TensorAdd {
    first: Box<Tensor>,
    second: Box<Tensor>,
}
impl TensorAdd {
    fn forward(mut input_a: Tensor, mut input_b: Tensor) -> Tensor {
        input_a.add_consumer();
        input_b.add_consumer();
        let result =
            input_a.container.borrow().array.clone() + input_b.container.borrow().array.clone();
        let node = TensorAdd {
            first: Box::new(input_a),
            second: Box::new(input_b),
        };
        Tensor::new_with_prev(result, Operation::Add(node))
    }
    fn backward(&mut self, output: &mut Tensor) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::zeros(output.shape()));
        let grad_a = grad.clone();
        let grad_b = grad.clone();
        self.first.backward_internal(grad_a);
        self.second.backward_internal(grad_b);
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
