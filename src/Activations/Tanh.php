<?php

namespace NumPower\Lattice\Activations;

use \NDArray as nd;

class Tanh extends Activation
{

    function activate(\NDArray $x): \NDArray
    {
        return nd::tanh($x);
    }

    function derivative(\NDArray $x): \NDArray
    {
        return 1 - nd::tanh($x) ** 2;
    }
}