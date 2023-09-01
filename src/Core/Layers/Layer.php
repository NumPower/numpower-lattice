<?php

namespace NumPower\Lattice\Core\Layers;

use NDArray as nd;
use NumPower\Lattice\Core\Initializers\IInitializer;
use NumPower\Lattice\Core\Regularizers\IRegularizer;
use NumPower\Lattice\Core\Variable;
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
     * @var int[]
     */
    protected array $inputShape;

    /**
     * @var bool
     */
    protected bool $trainable;

    /**
     * @var Variable[]
     */
    private array $trainableWeights = [];

    /**
     * @var Variable[]
     */
    private array $nonTrainableWeights = [];

    /**
     * @var ?string
     */
    private ?string $name;

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
        $this->inputShape = $shape;
    }

    /**
     * @return Variable[]
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
     * @return Variable
     */
    public function addWeight(
        ?string       $name = null,
        ?array        $shape = null,
        ?IInitializer $initializer = null,
        ?IRegularizer $regularizer = null,
        ?bool         $trainable = null
    ): Variable {
        $trainable == null || ($this->trainable = true);
        if ($initializer == null) {
            $initializer = new GlorotNormal();
        }
        $variable = new Variable(
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
     * @return int[]
     */
    public function getInputShape(): array
    {
        return $this->inputShape;
    }

    /**
     * @param Variable $inputs
     * @param bool $training
     * @return Variable
     */
    public function __invoke(Variable $inputs, bool $training = false): Variable
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
     * @return int[]
     */
    public function generateOutputShape(): array
    {
        return $this->inputShape;
    }
}
