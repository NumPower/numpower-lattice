<?php

namespace NumPower\Lattice\Layers;

use NumPower\Lattice\Activations\IActivation;

abstract class Layer implements ILayer
{
    private \NDArray $weights;

    private \NDArray $activation;

    /**
     * @var IActivation
     */
    private IActivation $activationFunction;

    private \NDArray $derivative;

    /**
     * @var bool
     */
    protected bool $isOutputLayer;

    public function __construct() {
        $this->isOutputLayer = false;
    }

    /**
     * @return \NDArray
     */
    public function weights(): \NDArray {
        return $this->weights;
    }

    /**
     * @return \NDArray
     */
    public function getWeights(): \NDArray
    {
        return $this->weights();
    }

    /**
     * @param \NDArray $weights
     * @return void
     */
    public function setWeights(\NDArray $weights): void
    {
        $this->weights = $weights;
    }

    /**
     * @param \NDArray $activation
     * @return void
     */
    protected function setActivation(\NDArray $activation): void
    {
        $this->activation = $activation;
    }

    /**
     * @return \NDArray
     */
    public function getActivation(): \NDArray
    {
        return $this->activation;
    }

    public function getActivationFunction(): IActivation
    {
        return $this->activationFunction;
    }

    public function setActivationFunction(IActivation $function): void
    {
        $this->activationFunction = $function;
    }

    public function setDerivative(\NDArray $derivative): void
    {
        $this->derivative = $derivative;
    }

    public function getDerivative(): \NDArray
    {
        return $this->derivative;
    }

    public function isOutput(): bool
    {
        return $this->isOutputLayer;
    }
}