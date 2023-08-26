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
        $ones = nd::ones($x->shape());
        $two = ($ones * 2);
        if ($x->isGPU()) {
            $two = $two->gpu();
            $ones = $ones->gpu();
        }
        return $ones - nd::tanh($x) ** $two;
    }
}