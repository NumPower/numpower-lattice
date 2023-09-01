<?php

namespace NumPower\Lattice\Core;

use NDArray as nd;
use NumPower\Lattice\Core\Initializers\IInitializer;
use NumPower\Lattice\Core\Regularizers\IRegularizer;
use NumPower\Lattice\Exceptions\ValueErrorException;

class Tensor
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
        string        $name,
        array         $shape,
        ?IInitializer $initializer = null,
        ?bool         $trainable = false,
        ?IRegularizer $regularizer = null,
        bool          $requireGrad = true
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
    private function initialize()
    {
        if ($this->initializer != null) {
            $this->array = ($this->initializer)($this->shape());
        }
    }

    /**
     * @param string $name
     * @param IGrad|null $callable
     * @return Operation
     */
    public function registerOperation(string $name, ?IGrad $callable = null): Operation
    {
        $op = new Operation($name, $this->inputs, $callable);
        $this->tape = $op;
        return $op;
    }

    /**
     * @param nd|float $array
     * @param string $name
     * @param bool $requireGrad
     * @return Tensor
     */
    public static function fromArray(\NDArray|float $array, string $name = "", bool $requireGrad = false): Tensor
    {
        if (is_float($array)) {
            $variable = new Tensor(
                name: $name,
                shape: []
            );
        } else {
            $variable = new Tensor(
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
        $this->grad = NULL;
        unset($this->tape);
    }

    /**
     * @param Tensor $a
     * @param Tensor $b
     * @return Tensor
     */
    public static function dot(Tensor $a, Tensor $b): Tensor
    {
        $new_var = Tensor::fromArray(nd::dot($a->getArray(), $b->getArray()));
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("dot");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @param array $newShape
     * @return Tensor
     */
    public static function reshape(Tensor $a, array $newShape): Tensor
    {
        $new_var = Tensor::fromArray(nd::reshape($a->getArray(), $newShape));
        $new_var->setInputs([$a, $newShape]);
        $new_var->registerOperation("reshape");
        return $new_var;
    }

    /**
     * @param Tensor|int|float $a
     * @param int|float|Tensor $b
     * @return Tensor
     */
    public static function maximum(Tensor|int|float $a, Tensor|int|float $b): Tensor
    {
        if (is_int($a) || is_float($a)) {
            $a = Tensor::fromArray($a);
        }
        if (is_int($b) || is_float($b)) {
            $b = Tensor::fromArray($b);
        }
        $new_var = Tensor::fromArray(nd::maximum($a->getArray(), $b->getArray()));
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("maximum");
        return $new_var;
    }

    private function updateShape(): void
    {
        $this->shape = $this->array->shape();
    }

    /**
     * @param Tensor|float|int $a
     * @param Tensor|float|int $b
     * @return Tensor
     */
    public static function divide(Tensor|float|int $a, Tensor|float|int $b): Tensor
    {
        if (is_int($a) || is_float($a)) {
            $a = Tensor::fromArray($a);
        }
        if (is_int($b) || is_float($b)) {
            $b = Tensor::fromArray($b);
        }
        $new_var = Tensor::fromArray($a->getArray() / $b->getArray());
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("divide");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @return Tensor
     */
    public static function abs(Tensor $a): Tensor
    {
        $new_var = Tensor::fromArray(nd::abs($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("abs");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @param Tensor $b
     * @return Tensor
     */
    public static function power(Tensor $a, Tensor $b): Tensor
    {
        $new_var = Tensor::fromArray($a->getArray() ** $b->getArray());
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("power");
        return $new_var;
    }

    /**
     * @return Tensor[]
     */
    public function getInputs(): array
    {
        return $this->inputs;
    }

    /**
     * @param Tensor $a
     * @param int $axis
     * @param bool $keepdim
     * @return Tensor
     */
    public static function sum_axis(Tensor $a, int $axis, bool $keepdim = false): Tensor
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
        $new_var = Tensor::fromArray($value);
        $new_var->setInputs([$a, $axis, $keepdim]);
        $new_var->registerOperation("sum_axis");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @param bool $keepdim
     * @return Tensor
     */
    public static function sum(Tensor $a, bool $keepdim = false): Tensor
    {
        $value = nd::sum($a->getArray());
        if ($keepdim) {
            if (is_float($value)) {
                $value = nd::ones($a->getArray()->shape()) * $value;
            } elseif (count($value->shape()) == 1 && count($value) == 1) {
                $value = $value[0] * nd::ones($a->getArray()->shape());
            }
        }
        $new_var = Tensor::fromArray($value);
        $new_var->setInputs([$a, $keepdim]);
        $new_var->registerOperation("sum");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @return Tensor
     */
    public static function log(Tensor $a): Tensor
    {
        $new_var = Tensor::fromArray(nd::log($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("log");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @return Tensor
     */
    public static function sqrt(Tensor $a): Tensor
    {
        $new_var = Tensor::fromArray(nd::sqrt($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("sqrt");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @param float $min
     * @param float $max
     * @return Tensor
     */
    public static function clip(Tensor $a, float $min, float $max): Tensor
    {
        $new_var = Tensor::fromArray(nd::clip($a->getArray(), $min, $max));
        $new_var->setInputs([$a, Tensor::fromArray($min), Tensor::fromArray($max)]);
        $new_var->registerOperation("clip");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @param Tensor $b
     * @return Tensor
     */
    public static function subtract(Tensor $a, Tensor $b): Tensor
    {
        $new_var = Tensor::fromArray($a->getArray() - $b->getArray());
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation("subtract");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @param Tensor $b
     * @return $this
     * @throws ValueErrorException
     */
    public static function add(Tensor $a, Tensor $b): Tensor
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
    public static function exp(Tensor $a): Tensor
    {
        $new_var = self::fromArray(nd::exp($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("exp");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @return Tensor
     */
    public static function mean(Tensor $a): Tensor
    {
        $new_var = Tensor::fromArray(nd::average($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("mean");
        return $new_var;
    }

    /**
     * @param Tensor $a
     * @param Tensor $b
     * @return Tensor
     */
    public static function multiply(Tensor $a, Tensor $b): Tensor
    {
        $new_var = Tensor::fromArray($a->getArray() * $b->getArray());
        $new_var->setInputs([$a, $b]);
        $new_var->registerOperation('multiply');
        return $new_var;
    }

    /**
     * @return ?Operation
     */
    public function getOperation(): ?Operation
    {
        if (!isset($this->tape)) {
            return null;
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
    public function diff(): \NDArray
    {
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
     * @param Tensor $a
     * @return Tensor
     */
    public static function tanh(Tensor $a): Tensor
    {
        $new_var = Tensor::fromArray(nd::tanh($a->getArray()));
        $new_var->setInputs([$a]);
        $new_var->registerOperation("tanh");
        return $new_var;
    }

    /**
     * @param nd|float|int|null $grad
     * @return void
     * @throws \Exception
     */
    public function backward(\NDArray|float|int $grad = null)
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

            if ($this->getOperation() != null) {
                $this->getOperation()->backward($grad);
            }
        }
    }
}
