<?php

namespace NumPower\Lattice\Core;

use \NDArray as nd;
use NumPower\Lattice\IGrad;

class Operation
{
    /**
     * @var string
     */
    private string $name;

    /**
     * @var Variable[]
     */
    private array $args;

    /**
     * @var Operation|null
     */
    private ?Operation $next;

    /**
     * @var IGrad|null
     */
    private ?IGrad $backwardFunction;

    /**
     * @param string $name
     * @param array $args
     * @param IGrad|null $backwardFunction
     */
    public function __construct(string $name, array $args, ?IGrad $backwardFunction = NULL) {
        $this->setName($name);
        $this->setArgs($args);
        $this->setBackwardFunction($backwardFunction);
    }

    /**
     * @param string $name
     * @return void
     */
    public function setName(string $name): void
    {
        $this->name = $name;
    }

    /**
     * @param array $args
     * @return void
     */
    public function setArgs(array $args): void
    {
        $this->args = $args;
    }

    /**
     * @return string
     */
    public function getName(): string
    {
        return $this->name;
    }

    /**
     * @return array
     */
    public function getArgs(): array
    {
        return $this->args;
    }

    /**
     * @param Operation $op
     * @return void
     */
    public function setNext(Operation $op): void {
        $this->next = $op;
    }

    /**
     * @param nd|float|int $grad
     * @return void
     * @throws \Exception
     */
    public function backward(\NDArray|float|int $grad): void
    {
        switch ($this->getName()) {
            case 'add':
                $this->getArgs()[0]->backward($grad);
                $this->getArgs()[1]->backward($grad);
                break;
            case 'power':
                $this->getArgs()[0]->backward($grad * $this->getArgs()[1]->getArray() * $this->getArgs()[0]->getArray() ** ($this->getArgs()[1]->getArray() - 1));
                $this->getArgs()[1]->backward($grad * $this->getArgs()[0]->getArray() ** $this->getArgs()[1]->getArray() * nd::log($this->getArgs()[0]->getArray()));
                break;
            case 'subtract':
                $this->getArgs()[0]->backward($grad);
                $this->getArgs()[1]->backward(-$grad);
                break;
            case 'divide':
                $this->getArgs()[0]->backward($grad / $this->getArgs()[1]->getArray());
                $this->getArgs()[1]->backward(-$grad * $this->getArgs()[0]->getArray() / $this->getArgs()[1]->getArray() ** 2);
                break;
            case 'exp':
                $this->getArgs()[0]->backward($grad * nd::exp($this->getArgs()[0]->getArray()));
                break;
            case 'tanh':
                $this->getArgs()[0]->backward($grad * (1 - (nd::tanh($this->getArgs()[0]->getArray()) ** 2)));
                break;
            case 'sqrt':
                $this->getArgs()[0]->backward($grad / (2 * nd::sqrt($this->getArgs()[0]->getArray())));
                break;
            case 'sum':
                $this->getArgs()[0]->backward(nd::ones($this->getArgs()[0]->getArray()->shape()) * $grad);
                break;
            case "log":
                $this->getArgs()[0]->backward($grad * (1 / $this->getArgs()[0]->getArray()));
                break;
            case 'sum_axis':
                // @todo Axis not supported
                $this->getArgs()[0]->backward(nd::ones($this->getArgs()[0]->getArray()->shape()) * $grad);
                break;
            case 'multiply':
                $this->getArgs()[0]->backward($grad * $this->getArgs()[1]->getArray());
                $this->getArgs()[1]->backward($this->getArgs()[0]->getArray() * $grad);
                break;
            case 'abs':
                $this->getArgs()[0]->backward($grad * nd::sign($this->getArgs()[0]->getArray()));
                break;
            case 'maximum':
                $this->getArgs()[0]->backward($grad * (nd::greater_equal($this->getArgs()[0]->getArray(), $this->getArgs()[1]->getArray())));
                $this->getArgs()[1]->backward($grad * (nd::less_equal($this->getArgs()[0]->getArray(), $this->getArgs()[1]->getArray())));
                break;
            case 'mean':
                $this->getArgs()[0]->backward((nd::ones($this->getArgs()[0]->getArray()->shape()) * $grad) / nd::prod(nd::array($this->args[0]->getArray()->shape())));
                break;
            case 'clip':
                $greater = nd::greater_equal($this->getArgs()[0]->getArray(), nd::ones($this->getArgs()[0]->getArray()->shape()) * $this->args[1]->getArray());
                $less = nd::less_equal($this->getArgs()[0]->getArray(), nd::ones($this->getArgs()[0]->getArray()->shape()) * $this->args[2]->getArray());
                $this->getArgs()[0]->backward($grad * $greater * $less);
                break;
            case 'dot':
                if (count($grad->shape()) == 1) {
                    $grad = nd::reshape($grad, [1, count($grad)]);
                }
                if (count($this->getArgs()[0]->getShape()) > 1 && count($this->getArgs()[0]->getShape()) > 1) {
                    $this->getArgs()[0]->backward(nd::dot($grad, nd::transpose($this->getArgs()[1]->getArray())));
                    $this->getArgs()[1]->backward(nd::dot(nd::transpose($this->getArgs()[0]->getArray()), $grad));
                } else {
                    throw new \Exception("Back propagation fatal error.");
                }
                break;
            default:
                if (!isset($this->backwardFunction)) {
                    throw new \Exception("Back propagation fatal error.");
                } else {
                    $this->backwardFunction->backward($grad, $this);
                }
        }
    }

    /**
     * @return Operation|null
     */
    public function getNext(): ?Operation
    {
        if (isset($this->next)) {
            return $this->next;
        }
        return NULL;
    }

    /**
     * @param IGrad|null $backwardFunction
     * @return void
     */
    private function setBackwardFunction(?IGrad $backwardFunction)
    {
        $this->backwardFunction = $backwardFunction;
    }
}