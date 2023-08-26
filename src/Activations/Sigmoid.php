<?php /** @noinspection PhpIncompatibleReturnTypeInspection */

namespace NumPower\Lattice\Activations;

use \NDArray as nd;

/**
 * The sigmoid activation function
 * 1 / (1 + exp(-x))
 */
class Sigmoid extends Activation
{
    /**
     * @param \NDArray $x
     * @return \NDArray
     */
    public function activate(\NDArray $x): \NDArray {
        $ones = nd::ones($x->shape());
        if ($x->isGPU()) {
            $ones = $ones->gpu();
        }
        return $ones / (nd::exp(nd::negative($x)) + $ones);
    }

    /**
     * @param nd $y
     * @return nd
     */
    function derivative(\NDArray $x): \NDArray
    {
        $ones = nd::ones($x->shape());
        if ($x->isGPU()) {
            $ones = $ones->gpu();
        }
        return $x * ($ones - $x);
    }
}