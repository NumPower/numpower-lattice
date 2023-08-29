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
     * @var Tape
     */
    private Tape $tape;

    /**
     * @param string $name
     * @param array $shape
     * @param IInitializer|null $initializer
     * @param bool $trainable
     * @param IRegularizer|null $regularizer
     */
    public function __construct(
        string $name,
        array $shape,
        ?IInitializer $initializer = NULL,
        ?bool $trainable = False,
        ?IRegularizer $regularizer = NULL
    )
    {
        $this->tape = new Tape();
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
     * @param array $args
     * @return Operation
     * @throws ValueErrorException
     */
    public function registerOperation(string $name, array $args): Operation
    {
        $op = new Operation($name, $args);
        $this->tape->add($op);
        return $op;
    }

    /**
     * @param string $name
     * @param nd $array
     * @return Variable
     */
    public static function fromArray(string $name, \NDArray $array): Variable {
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
     * @return void
     */
    public function clearTape(): void
    {
        $this->tape = [
            $this->tape[0]
        ];
    }

    /**
     * @param nd $a
     * @return Variable
     * @throws ValueErrorException
     */
    public function dot(\NDArray $a): Variable
    {
        $this->registerOperation("dot", [$a]);
        $this->array = nd::dot($this->array, $a);
        $this->updateShape();
        return $this;
    }

    private function updateShape(): void
    {
        $this->shape = $this->array->shape();
    }

    /**
     * @param nd $a
     * @return Variable
     * @throws ValueErrorException
     */
    public function denominatorDivide(\NDArray $a): Variable
    {
        $this->registerOperation("denominator_divide", [$a]);
        $this->array = $a / $this->array;
        $this->updateShape();
        return $this;
    }

    /**
     * @param nd $a
     * @return $this
     * @throws ValueErrorException
     */
    public function add(\NDArray $a): Variable
    {
        $this->registerOperation("add", [$a]);
        $this->array = nd::add($this->array, $a);
        $this->updateShape();
        return $this;
    }

    /**
     * @return $this
     * @throws ValueErrorException
     */
    public function exp(): Variable
    {
        $this->registerOperation("exp", []);
        $this->array = nd::exp($this->array);
        return $this;
    }

    /**
     * @return $this
     * @throws ValueErrorException
     */
    public function negative(): Variable
    {
        $this->registerOperation("negative", []);
        $this->array = nd::negative($this->array);
        return $this;
    }

    /**
     * @return Tape
     */
    public function getTape(): Tape
    {
        return $this->tape;
    }

    /**
     * @param $target
     * @param $args
     * @return nd
     */
    private static function _dot_derivative($target, $args): \NDArray
    {
        return $args[0];
    }

    /**
     * @param $target
     * @param $args
     * @return nd
     */
    private static function _add_derivative($target, $args): \NDArray
    {
        return nd::ones($args[0]->shape());
    }

    /**
     * @param $target
     * @param $args
     * @return nd
     */
    private static function _negative_derivative($target, $args): \NDArray
    {
        return -nd::ones($target->shape());
    }

    /**
     * @param $target
     * @param $args
     * @return nd
     */
    private static function _exp_derivative($target, $args): \NDArray
    {
        return nd::exp($target);
    }

    /**
     * @param $target
     * @param $args
     * @return nd
     */
    private static function _denominator_divide_derivative($target, $args): \NDArray
    {
        return (1 / $args[0]) - ($target / ($args[0] ** 2));
    }

    /**
     * @param int $op_index
     * @return nd
     */
    public function partialDiff(int $op_index): \NDArray
    {
        $current_val = $this->getArray();
        $tape_head = $this->getTape()->getHead();
        for($i = count($this->getTape()) - 1; $i >= $op_index; $i--) {
            $op = $tape_head;
            $current_val = call_user_func(self::class . "::_" .$op->getName()."_derivative", $current_val, $op->getArgs());
            $tape_head = $tape_head->getNext();
        }
        return $current_val;
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
     * @return nd
     */
    public function diff(): \NDArray {
        $current_val = $this->getArray();
        $tape_head = $this->getTape()->getHead();
        while(isset($tape_head)) {
            $current_val = call_user_func(self::class . "::_" .$tape_head->getName()."_derivative", $current_val, $tape_head->getArgs());
            $tape_head = $tape_head->getNext();
        }
        return $current_val;
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
        return $this->shape();
    }

    /**
     * @return Variable
     * @throws ValueErrorException
     */
    public function tanh(): Variable
    {
        $this->registerOperation("tanh", []);
        $this->array = nd::tanh($this->array);
        return $this;
    }
}