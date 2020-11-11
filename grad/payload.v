module grad

// A Payload is a simple wrapper around a Variable.  It
// is only abstracted out to be a bit more explicit that
// it is being passed around through an operation
pub struct Payload {
pub:
	// Contents of the paylod
	variable &Variable
}

pub fn new_payload(variable &Variable) &Payload {
	return &Payload{
		variable: variable
	}
}
