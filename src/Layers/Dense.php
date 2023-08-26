<?php

namespace NumPower\Lattice\Layers;

use \NDArray as nd;
use NumPower\Lattice\Activations\IActivation;
use NumPower\Lattice\Initializers\IInitializer;
use NumPower\Lattice\Initializers\RandomNormal;
use NumPower\Lattice\Initializers\RandomUniform;
use NumPower\Lattice\Initializers\Zeros;

class Dense extends Layer
{
    /**
     * @var int
     */
    private int $units;

    /**
     * @var IInitializer|null
     */
    private ?IInitializer $weightsInitializer;

    /**
     * @var array
     */
    private array $inputShape;

    /**
     * @var array
     */
    private array $outputShape;

    /**
     * @param int $units
     * @param IActivation $activation
     * @param IInitializer|null $weightsInitializer
     */
    public function __construct(int $units, IActivation $activation, IInitializer $weightsInitializer = NULL) {
        $this->units = $units;
        $this->weightsInitializer = $weightsInitializer;
        $this->setActivationFunction($activation);
        parent::__construct();
    }

    /**
     * @return array
     */
    public function inputShape(): array
    {
        return $this->inputShape;
    }

    /**
     * @param array $inputShape
     * @return void
     */
    private function setInputShape(array $inputShape): void {
        $this->inputShape = $inputShape;
    }

    /**
     * @return array
     */
    public function generateOutputShape(): array
    {
        if ($this->isOutputLayer) {
            return [$this->units];
        }
        if (count($this->inputShape) == 1) {
            return [$this->inputShape[0], $this->units];
        }
        return [$this->inputShape[1], $this->units];
    }

    /**
     * @return void
     */
    private function initializeWeights(bool $use_gpu): void
    {
        if(!isset($this->weightsInitializer)) {
            if (!$this->isOutputLayer) {
                $this->weightsInitializer = new RandomUniform();
                $this->weightsInitializer->setShape($this->generateOutputShape());
            } else {
                $this->weightsInitializer = new RandomUniform();
                $this->weightsInitializer->setShape([$this->inputShape[count($this->inputShape) - 1], $this->units]);
            }
        }
        $this->setWeights($this->weightsInitializer->initialize($use_gpu));
    }

    /**
     * @param array $shape
     * @return void
     */
    private function setOutputShape(array $shape): void
    {
        $this->outputShape = $shape;
    }

    /**
     * @param ILayer $previousLayer
     * @param bool $use_gpu
     * @param bool $isOutput
     * @return void
     */
    public function initialize(ILayer $previousLayer, bool $use_gpu, bool $isOutput = false): void
    {
        $this->isOutputLayer = $isOutput;
        $this->setInputShape($previousLayer->generateOutputShape());
        $this->initializeWeights($use_gpu);
    }

    /**
     * Forward Propagation
     *
     * @param nd $inputs
     * @return nd
     */
    public function forward(\NDArray $inputs): \NDArray {
        $net_input = nd::dot($inputs, $this->weights());
        $activations = $this->getActivationFunction()->activate($net_input);
        $this->setActivation($activations);
        return $activations;
    }

    /**
     * @param ILayer $previousLayer
     * @param nd $error
     * @return nd
     */
    public function backward(ILayer $previousLayer, \NDArray $error): \NDArray {
        $previous_activation = $previousLayer->getActivation();
        if (count($error->shape()) == 1) {
            $delta = nd::multiply(nd::reshape($error, [1, count($error)]), $previousLayer->getActivationFunction()->derivative($previous_activation));
            //$delta = nd::transpose($delta);
        } else {
            $delta = nd::multiply($error, $previousLayer->getActivationFunction()->derivative($previous_activation));
        }
        $this->setDerivative(
            nd::dot(nd::transpose($this->getActivation()), $delta)
        );
        return nd::dot($delta, nd::transpose($previousLayer->weights()));
    }
}