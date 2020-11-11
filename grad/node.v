module grad

// A Node is a member of a computational graph that contains
// a reference to a gate, as well as the parents of the operation
// and the payload that resulted from the operation.
pub struct Node {
pub:
	// A Gate containing a backwards and cache function for
	// a node
	gate    &Gate
	// The variables that created this node
	parents []Variable
	// Wrapper around a Tensor, contains operation data
	payload &Payload
	// Debug use only, contains a name for a node
	name    string
}

// This initializer shouldn't ever be called outside of
// vnum.grad.register.
// Users defining custom gradients and gates should
// follow the same rule
pub fn new_node(gate &Gate, parents []Variable, payload &Payload, name string) &Variable {
	return &Variable{
		gate: gate
		parents: parents
		payload: payload
		name: name
	}
}
