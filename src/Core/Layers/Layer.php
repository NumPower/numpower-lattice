<?php

namespace NumPower\Lattice\Core\Layers;

use NDArray as nd;
use NumPower\Lattice\Core\Initializers\IInitializer;
use NumPower\Lattice\Core\Regularizers\IRegularizer;
use NumPower\Lattice\Core\Tensor;
use NumPower\Lattice\Exceptions\ValueErrorException;
use NumPower\Lattice\Initializers\GlorotNormal;
use NumPower\Lattice\Utils\LayerUtils;

class Layer implements ILayer
{
    /**
     * @var bool
     */
    protected bool $built;

    /**
     * @var bool
     */
    protected bool $trainable;

    /**
     * @var Tensor[]
     */
    private array $trainableWeights = [];

    /**
     * @var Tensor[]
     */
    private array $nonTrainableWeights = [];

    /**
     * @var ?string
     */
    private ?string $name;

    /**
     * @var bool
     */
    private bool $supportMasking;

    /**
     */
    private InputSpec $inputSpec;

    /**
     * @var int
     */
    private int $batchSize;

    /**
     * @param string $name
     * @param bool $trainable
     */
    public function __construct(string $name, bool $trainable = false)
    {
        $this->built = false;
        $this->trainable = $trainable;
        $this->setName($name);
    }

    /**
     * @param int[] $shape
     * @return void
     */
    public function setInputShape(array $shape): void
    {
        $this->inputSpec = new InputSpec(
            shape: $shape
        );
    }

    /**
     * @param bool $value
     * @return void
     */
    protected function setSupportMasking(bool $value): void
    {
        $this->supportMasking = $value;
    }

    /**
     * @return bool
     */
    protected function getSupportMasking(): bool
    {
        return $this->supportMasking;
    }

    /**
     * @return Tensor[]
     */
    public function getTrainableWeights(): array
    {
        return $this->trainableWeights;
    }

    /**
     * @param bool $built
     * @return void
     */
    public function setBuilt(bool $built): void
    {
        $this->built = $built;
    }

    /**
     * @return bool
     */
    public function built(): bool
    {
        return $this->built;
    }

    /**
     * @param ?string $name
     * @return void
     */
    public function setName(?string $name): void
    {
        $this->name = $name;
    }

    /**
     * @return string
     */
    public function getName(): string
    {
        return $this->name;
    }

    /**
     * @return bool
     */
    public function isTrainable(): bool
    {
        return $this->trainable;
    }

    /**
     * @return int
     * @throws ValueErrorException
     */
    public function countParams(): int
    {
        if (!$this->built()) {
            throw new ValueErrorException("Tried to call countParams on layer " . $this->getName() . ", but the layer isn´t built");
        }
        return LayerUtils::countParams(array_merge($this->trainableWeights, $this->nonTrainableWeights));
    }

    /**
     * @return int
     * @throws ValueErrorException
     */
    public function countTrainableParams(): int
    {
        if (!$this->built()) {
            throw new ValueErrorException("Tried to call countTrainableParams on layer " . $this->getName() . ", but the layer isn´t built");
        }
        return LayerUtils::countParams($this->trainableWeights);
    }

    /**
     * @return int
     * @throws ValueErrorException
     */
    public function countNonTrainableParams(): int
    {
        if (!$this->built()) {
            throw new ValueErrorException("Tried to call countNonTrainableParams on layer " . $this->getName() . ", but the layer isn´t built");
        }
        return LayerUtils::countParams($this->nonTrainableWeights);
    }

    /**
     * Add a variable to the layer
     *
     * @param string|null $name
     * @param int[]|null $shape
     * @param IInitializer|null $initializer
     * @param IRegularizer|null $regularizer
     * @param bool|null $trainable
     * @return Tensor
     */
    public function addWeight(
        ?string       $name = null,
        ?array        $shape = null,
        ?IInitializer $initializer = null,
        ?IRegularizer $regularizer = null,
        ?bool         $trainable = null
    ): Tensor
    {
        $trainable == null || ($this->trainable = true);
        if ($initializer == null) {
            $initializer = new GlorotNormal();
        }
        $variable = new Tensor(
            name: $name,
            shape: $shape,
            initializer: $initializer,
            trainable: $trainable,
            regularizer: $regularizer,
            requireGrad: true
        );

        if ($trainable) {
            $this->trainableWeights[] = $variable;
        } else {
            $this->nonTrainableWeights[] = $variable;
        }

        return $variable;
    }

    /**
     * @return InputSpec
     */
    public function getInputSpec(): InputSpec
    {
        return $this->inputSpec;
    }

    /**
     * @param InputSpec $spec
     * @return void
     */
    public function setInputSpec(InputSpec $spec): void
    {
        $this->inputSpec = $spec;
    }

    /**
     * @param int $batchSize
     * @return void
     */
    public function setBatchSize(int $batchSize): void
    {
        $this->batchSize = $batchSize;
    }

    /**
     * @return int
     */
    public function getBatchSize(): int
    {
        return $this->batchSize;
    }

    /**
     * @return int[]
     */
    public function getInputShape(): array
    {
        return $this->inputSpec->getShape();
    }

    /**
     * @param Tensor $inputs
     * @param bool $training
     * @return Tensor
     */
    public function __invoke(Tensor $inputs, bool $training = false): Tensor
    {
        return $inputs;
    }

    /**
     * @param int[] $inputShape
     * @return void
     */
    public function build(array $inputShape): void
    {
        $this->built = true;
    }

    /**
     * @return array
     */
    public function generateOutputShape(): array
    {
        return $this->inputSpec->getShape();
    }
}
