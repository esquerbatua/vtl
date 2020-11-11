module grad

import num

// A Gate is an object that can cache the result of an operation,
// as well as backpropogate a payload backwards along the
// computational graph
//
// Child classes that inherit from this class can add instance
// variables if additional caching is needed, and these need
// to be populated when writing the cached operation
pub interface Gate {
  // Propogates an operation backwards, transforming a payload
  // and returning an array of Tensors
  backward(payload Payload) []num.NdArray

  // Caches the result of an operation on a context
  cache(result Variable, args ...any)
}
