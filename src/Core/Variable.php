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
     * @var \NDArray|float
     */
    private \NDArray|float $array;

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
     * @var float|nd|null
     */
    private \NDArray|float|null $grad;

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
     * @return nd|float
     */
    public function getArray(): \NDArray|float
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
     * @param nd|float $array
     * @param string $name
     * @param bool $requireGrad
     * @return Variable
     */
    public static function fromArray(\NDArray|float $array, string $name = "", bool $requireGrad = False): Variable {
        if (is_float($array)) {
            $variable = new Variable(
                name: $name,
                shape: []
            );
        } else {
            $variable = new Variable(
                name: $name,
                shape: $array->shape()
            );
        }
        $variable->overwriteArray($array);
        return $variable;
    }

    /**
     * @param nd|float $array
     * @return void
     */
    public function overwriteArray(\NDArray|float $array): void
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
     * @return Variable
     */
    public static function abs(Variable $a): Variable {
        $new_var = Variable::fromArray(nd::abs($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("abs");
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
     * @param int $axis
     * @param bool $keepdim
     * @return Variable
     */
    public static function sum_axis(Variable $a, int $axis, bool $keepdim = False): Variable
    {
        $value = nd::sum($a->getArray(), $axis);
        if ($keepdim) {
            if (count($value->shape()) == 1 && count($value) == 1) {
                $value = $value[0] * nd::ones($a->getArray()->shape());
            }
            if (count($value->shape()) == 1 && count($a->getArray()->shape()) == 2) {
                $value = nd::reshape($value, [count($value), 1]);
            }
        }
        $new_var = Variable::fromArray($value);
        $new_var->setInputs([$a, $axis, $keepdim]);
        $new_var->registerOperation("sum_axis");
        return $new_var;
    }

    /**
     * @param Variable $a
     * @param bool $keepdim
     * @return Variable
     */
    public static function sum(Variable $a, bool $keepdim = False): Variable
    {
        $value = nd::sum($a->getArray());
        if ($keepdim) {
            if (is_float($value)) {
                $value = nd::ones($a->getArray()->shape()) * $value;
            } elseif (count($value->shape()) == 1 && count($value) == 1) {
                $value = $value[0] * nd::ones($a->getArray()->shape());
            }
        }
        $new_var = Variable::fromArray($value);
        $new_var->setInputs([$a, $keepdim]);
        $new_var->registerOperation("sum");
        return $new_var;
    }

    /**
     * @param Variable $a
     * @return Variable
     */
    public static function log(Variable $a): Variable
    {
        $new_var = Variable::fromArray(nd::log($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("log");
        return $new_var;
    }

    /**
     * @param Variable $a
     * @return Variable
     */
    public static function sqrt(Variable $a): Variable
    {
        $new_var = Variable::fromArray(nd::sqrt($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("sqrt");
        return $new_var;
    }

    /**
     * @param Variable $a
     * @param float $min
     * @param float $max
     * @return Variable
     */
    public static function clip(Variable $a, float $min, float $max): Variable
    {
        $new_var = Variable::fromArray(nd::clip($a->getArray(), $min, $max));
        $new_var->setInputs([$a, Variable::fromArray($min), Variable::fromArray($max)]);
        $new_var->registerOperation("clip");
        return $new_var;
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
     * @return Variable
     */
    public static function mean(Variable $a): Variable
    {
        $new_var = Variable::fromArray(nd::average($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("mean");
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
     * @param nd|float|int|null $grad
     * @return void
     * @throws \Exception
     */
    public function backward(\NDArray|float|int $grad = NULL)
    {
        if (!isset($grad)) {
            if (!is_float($this->getArray()) && !is_int($this->getArray())) {
                $grad = nd::ones($this->getArray()->shape());
            } else {
                $grad = 1;
            }
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