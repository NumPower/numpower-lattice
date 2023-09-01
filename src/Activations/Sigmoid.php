<?php /** @noinspection PhpIncompatibleReturnTypeInspection */

namespace NumPower\Lattice\Activations;

use NDArray as nd;
use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Tensor;
use NumPower\Lattice\Exceptions\ValueErrorException;

/**
 * The sigmoid activation function
 * 1 / (1 + exp(-x))
 */
class Sigmoid implements IActivation
{
    /**
     * @param Tensor $inputs
     * @return Tensor
     * @throws ValueErrorException
     */
    public function __invoke(Tensor $inputs): Tensor
    {
        $minus_ones = Tensor::fromArray(-1);
        $ones = Tensor::fromArray(1);
        return Tensor::divide(
            $ones,
            Tensor::add(
                Tensor::exp(
                    Tensor::multiply($inputs, $minus_ones)
                ),
                $ones
            )
        );
    }
}
