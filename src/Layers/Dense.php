<?php

namespace NumPower\Lattice\Layers;

use NDArray as nd;
use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Initializers\IInitializer;
use NumPower\Lattice\Core\Layers\Layer;
use NumPower\Lattice\Core\Regularizers\IRegularizer;
use NumPower\Lattice\Core\Tensor;
use NumPower\Lattice\Exceptions\ValueErrorException;
use NumPower\Lattice\Initializers\Zeros;

class Dense extends Layer
{
    /**
     * @var ?IActivation
     */
    private ?IActivation $activation;

    /**
     * @var bool
     */
    private bool $useBias;

    /**
     * @var IInitializer|null
     */
    private ?IInitializer $kernelInitializer;

    /**
     * @var IInitializer|null
     */
    private ?IInitializer $biasInitializer;

    /**
     * @var int
     */
    private int $units;

    /**
     * @var IRegularizer|null
     */
    private ?IRegularizer $kernelRegularizer;

    /**
     * @var IRegularizer|null
     */
    private ?IRegularizer $biasRegularizer;

    /**
     * @var Tensor
     */
    private Tensor $kernel;

    /**
     * @var Tensor
     */
    private Tensor $bias;

    /**
     * @param int $units
     * @param IActivation|null $activation
     * @param bool $useBias
     * @param IInitializer|null $kernelInitializer
     * @param IInitializer|null $biasInitializer
     * @param IRegularizer|null $kernelRegularizer
     * @param IRegularizer|null $biasRegularizer
     */
    public function __construct(
        int           $units,
        ?IActivation  $activation = null,
        bool          $useBias = true,
        ?IInitializer $kernelInitializer = null,
        ?IInitializer $biasInitializer = null,
        ?IRegularizer $kernelRegularizer = null,
        ?IRegularizer $biasRegularizer = null
    ) {
        parent::__construct("dense_" . substr(uniqid(), -4), true);
        $this->units = $units;
        $this->activation = $activation;
        $this->useBias = $useBias;
        $this->kernelInitializer = $kernelInitializer;
        $this->biasInitializer = $biasInitializer;
        $this->kernelRegularizer = $kernelRegularizer;
        $this->biasRegularizer = $biasRegularizer;
    }

    /**
     * @param array $inputShape
     * @return void
     */
    public function build(array $inputShape): void
    {
        $this->setInputShape($inputShape);
        $last_dim = $inputShape[count($inputShape) - 1];
        $this->kernel = $this->addWeight(
            name: "kernel",
            shape: [$last_dim, $this->units],
            initializer: $this->kernelInitializer,
            regularizer: $this->kernelRegularizer,
            trainable: true
        );
        if ($this->useBias) {
            if (!isset($this->biasInitializer)) {
                $this->biasInitializer = new Zeros();
            }
            $this->bias = $this->addWeight(
                name: "bias",
                shape: [$this->units],
                initializer: $this->biasInitializer,
                regularizer: $this->biasRegularizer,
                trainable: true
            );
        }
        $this->setBuilt(true);
    }

    /**
     * @param Tensor $inputs
     * @param bool $training
     * @return Tensor
     * @throws ValueErrorException
     */
    public function __invoke(Tensor $inputs, bool $training = false): Tensor
    {
        if (!$this->built()) {
            $this->build($inputs->shape());
        }
        $outputs = Tensor::dot($inputs, $this->kernel);
        if ($this->useBias) {
            $outputs = Tensor::add($outputs, $this->bias);
        }
        if ($this->activation) {
            $outputs = ($this->activation)($outputs);
        }
        return $outputs;
    }

    /**
     * @return array
     */
    public function generateOutputShape(): array
    {
        $shape = $this->getInputShape();
        assert (count($shape) == 2);
        array_pop($shape);
        $shape = [$this->getInputShape()[0], $this->units];
        return $shape;
    }
}
