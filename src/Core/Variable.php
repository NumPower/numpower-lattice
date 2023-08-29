<?php

namespace NumPower\Lattice\Core;

use \NDArray as nd;
use NumPower\Lattice\Core\Initializers\IInitializer;
use NumPower\Lattice\Core\Regularizers\IRegularizer;
use NumPower\Lattice\Exceptions\ValueErrorException;

class Variable
{
    /**
     * @var ?string
     */
    private ?string $name;

    /**
     * @var bool
     */
    private bool $trainable;

    /**
     * @var IInitializer|null
     */
    private ?IInitializer $initializer;

    /**
     * @var array
     */
    private array $shape;

    /**
     * @var IRegularizer|null
     */
    private ?IRegularizer $regularizer;

    /**
     * @var \NDArray
     */
    private \NDArray $array;

    /**
     * @var Operation
     */
    private Operation $tape;

    /**
     * @var array
     */
    private array $inputs;

    /**
     * @var bool
     */
    private bool $requireGrad;

    /**
     * @var nd|null
     */
    private ?\NDArray $grad;

    /**
     * @param string $name
     * @param array $shape
     * @param IInitializer|null $initializer
     * @param bool $trainable
     * @param IRegularizer|null $regularizer
     * @param bool $requireGrad
     */
    public function __construct(
        string $name,
        array $shape,
        ?IInitializer $initializer = NULL,
        ?bool $trainable = False,
        ?IRegularizer $regularizer = NULL,
        bool $requireGrad = True
    )
    {
        $this->requireGrad = $requireGrad;
        $this->name = $name;
        $this->shape = $shape;
        $this->trainable = $trainable;
        $this->initializer = $initializer;
        $this->regularizer = $regularizer;
        $this->initialize();
    }

    /**
     * @return nd
     */
    public function getArray(): \NDArray
    {
        return $this->array;
    }

    /**
     * @return void
     */
    private function initialize() {
        if ($this->initializer != NULL) {
            $this->array = ($this->initializer)($this->shape());
        }
    }

    /**
     * @param string $name
     * @return Operation
     */
    public function registerOperation(string $name): Operation
    {
        $op = new Operation($name, $this->inputs);
        $this->tape = $op;
        return $op;
    }

    /**
     * @param string $name
     * @param nd $array
     * @return Variable
     */
    public static function fromArray(\NDArray $array, string $name = "", bool $requireGrad = False): Variable {
        $variable = new Variable(
            name: $name,
            shape: $array->shape()
        );
        $variable->overwriteArray($array);
        return $variable;
    }

    /**
     * @param nd $array
     * @return void
     */
    public function overwriteArray(\NDArray $array): void
    {
        $this->array = $array;
    }

    /**
     * @param Variable $a
     * @param Variable $b
     * @return Variable
     */
    public static function dot(Variable $a, Variable $b): Variable
    {
        $new_var = Variable::fromArray(nd::dot($a->getArray(), $b->getArray()));
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("dot");
        return $new_var;
    }

    private function updateShape(): void
    {
        $this->shape = $this->array->shape();
    }

    /**
     * @param Variable $a
     * @param Variable $b
     * @return Variable
     */
    public static function divide(Variable $a, Variable $b): Variable
    {
        $new_var = Variable::fromArray( $a->getArray() / $b->getArray());
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("divide");
        return $new_var;
    }

    /**
     * @param Variable $a
     * @param Variable $b
     * @return Variable
     */
    public static function power(Variable $a, Variable $b): Variable
    {
        $new_var = Variable::fromArray( $a->getArray() ** $b->getArray());
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("power");
        return $new_var;
    }

    /**
     * @return Variable[]
     */
    public function getInputs(): array
    {
        return $this->inputs;
    }

    /**
     * @param Variable $a
     * @param Variable $b
     * @return Variable
     */
    public static function subtract(Variable $a, Variable $b): Variable
    {
        $new_var = Variable::fromArray( $a->getArray() - $b->getArray());
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("subtract");
        return $new_var;
    }

    /**
     * @param Variable $a
     * @param Variable $b
     * @return $this
     * @throws ValueErrorException
     */
    public static function add(Variable $a, Variable $b): Variable
    {
        $new_var = self::fromArray(nd::add($a->getArray(), $b->getArray()));
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("add");
        return $new_var;
    }

    /**
     * @param array $val
     * @return void
     */
    public function setInputs(array $val): void
    {
        $this->inputs = $val;
    }

    /**
     * @return $this
     * @throws ValueErrorException
     */
    public static function exp(Variable $a): Variable
    {
        $new_var = self::fromArray(nd::exp($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("exp");
        return $new_var;
    }

    /**
     * @param Variable $a
     * @param Variable $b
     * @return Variable
     */
    public static function multiply(Variable $a, Variable $b): Variable {
        $new_var = Variable::fromArray($a->getArray() * $b->getArray());
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation('multiply');
        return $new_var;
    }

    /**
     * @return ?Operation
     */
    public function getOperation(): ?Operation
    {
        if (!isset($this->tape)){
            return NULL;
        }
        return $this->tape;
    }

    /**
     * @param $target
     * @param $args
     * @return nd
     */
    private static function _tanh_derivative($target, $args): \NDArray
    {
        $tanh_x = nd::tanh($target);
        return 1 - $tanh_x ** 2;
    }

    /**
     * @return string
     */
    public function getName(): string
    {
        return $this->name;
    }

    /**
     * @return nd
     */
    public function diff(): \NDArray {
        return $this->grad;
    }

    /**
     * @return array
     */
    public function shape(): array
    {
        return $this->shape;
    }

    /**
     * @return array
     */
    public function getShape(): array
    {
        return $this->getArray()->shape();
    }

    /**
     * @param Variable $a
     * @return Variable
     */
    public static function tanh(Variable $a): Variable
    {
        $new_var = Variable::fromArray(nd::tanh($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("tanh");
        return $new_var;
    }

    /**
     * @return void
     */
    public function backward(?\NDArray $grad = NULL)
    {
        if (!isset($grad)) {
            $grad = nd::ones($this->getArray()->shape());
        }

        if ($this->requireGrad) {
            if (!isset($this->grad)) {
                $this->grad = $grad;
            } else {
                $this->grad += $grad;
            }

            if ($this->getOperation() != NULL) {
                $this->getOperation()->backward($grad);
            }
        }
    }
}