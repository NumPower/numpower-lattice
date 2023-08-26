<?php

namespace NumPower\Lattice\Layers;

use \NDArray as nd;

class Input extends Layer
{
    /**
     * @var array
     */
    private array $inputShape;

    /**
     * @param array $inputShape
     */
    public function __construct(array $inputShape) {
        $this->inputShape = $inputShape;
        parent::__construct();
    }

    public function inputShape(): array
    {
        return [$this->inputShape];
    }

    public function generateOutputShape(): array
    {
        return $this->inputShape;
    }

    public function initialize(ILayer $nextLayer, bool $use_gpu, bool $isOutput = false): void
    {
        // TODO: Implement initialize() method.
    }

    /**
     * Forward Propagation
     *
     * @param \NDArray $inputs
     * @return \NDArray
     */
    public function forward(\NDArray $inputs): \NDArray {
        $this->setActivation($inputs);
        return $inputs;
    }

    /**
     * @param ILayer $previousLayer
     * @param \NDArray $error
     * @return void
     */
    function backward(ILayer $previousLayer, \NDArray $error): \NDArray
    {
        $previous_activation = $previousLayer->getActivation();
        if (count($error->shape()) == 1) {
            $delta = nd::multiply(nd::reshape($error, [1, count($error)]), $previousLayer->getActivationFunction()->derivative($previous_activation));
            $delta = nd::transpose($delta);
        } else {
            $delta = nd::multiply($error, $previousLayer->getActivationFunction()->derivative($previous_activation));
        }

        $this->setDerivative(
            nd::dot(nd::transpose($this->getActivation()), $delta)
        );
        return nd::dot($delta, nd::transpose($previousLayer->weights()));
    }
}