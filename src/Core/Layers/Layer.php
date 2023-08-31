<?php

namespace NumPower\Lattice\Core\Layers;

use NumPower\Lattice\Core\Initializers\IInitializer;
use NumPower\Lattice\Core\Regularizers\IRegularizer;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Exceptions\ValueErrorException;
use NumPower\Lattice\Initializers\RandomNormal;
use NumPower\Lattice\Initializers\RandomUniform;
use NumPower\Lattice\Utils\LayerUtils;

class Layer implements ILayer
{
    /**
     * @var bool
     */
    protected bool $built;

    /**
     * @var array
     */
    protected array $inputShape;

    /**
     * @var bool
     */
    protected bool $trainable;

    /**
     * @var array
     */
    private array $trainableWeights = [];

    /**
     * @var array
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
    public function __construct(string $name, bool $trainable = False) {
        $this->built = False;
        $this->trainable = $trainable;
        $this->setName($name);
    }

    /**
     * @param array $shape
     * @return void
     */
    public function setInputShape(array $shape): void {
        $this->inputShape = $shape;
    }

    /**
     * @return array
     */
    public function getTrainableWeights(): array
    {
        return $this->trainableWeights;
    }

    /**
     * @param bool $built
     * @return bool
     */
    public function setBuilt(bool $built) {
        return $this->built = $built;
    }

    /**
     * @return bool
     */
    public function built(): bool {
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
     * @param array|null $shape
     * @param IInitializer|null $initializer
     * @param IRegularizer|null $regularizer
     * @param bool|null $trainable
     * @return Variable
     */
    public function addWeight(
        ?string $name = NULL,
        ?array $shape = NULL,
        ?IInitializer $initializer = NULL,
        ?IRegularizer $regularizer = NULL,
        ?bool $trainable = NULL
    ): Variable {
        $trainable == NULL || ($this->trainable = True);
        if ($initializer == NULL) {
            $initializer = new RandomUniform();
        }
        $variable = new Variable(
            name: $name,
            shape: $shape,
            initializer: $initializer,
            trainable: $trainable,
            regularizer: $regularizer,
            requireGrad: True
        );

        if ($trainable) {
            $this->trainableWeights[] = $variable;
        } else {
            $this->nonTrainableWeights[] = $variable;
        }

        return $variable;
    }

    /**
     * @return array
     */
    public function getInputShape(): array
    {
        return $this->inputShape;
    }

    /**
     * @param Variable $inputs
     * @return Variable
     */
    function __invoke(Variable $inputs): Variable
    {
        return $inputs;
    }

    /**
     * @param array $inputShape
     * @return void
     */
    function build(array $inputShape)
    {
        $this->built = True;
    }

    /**
     * @return array
     */
    function generateOutputShape(): array
    {
        return $this->inputShape;
    }
}