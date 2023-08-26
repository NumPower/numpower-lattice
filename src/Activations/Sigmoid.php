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
        return 1 / (nd::exp(-$x) + 1);
    }

    /**
     * @param nd $y
     * @return nd
     */
    function derivative(\NDArray $x): \NDArray
    {
        return $x * (1 - $x);
    }
}