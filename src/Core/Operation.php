<?php

namespace NumPower\Lattice\Core;

use \NDArray as nd;

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

    public function __construct(string $name, array $args) {
        $this->setName($name);
        $this->setArgs($args);
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
                $this->args[0]->backward($grad);
                $this->args[1]->backward($grad);
                break;
            case 'power':
                $this->args[0]->backward($grad * $this->args[1]->getArray() * $this->args[0]->getArray() ** ($this->args[1]->getArray() - 1));
                $this->args[1]->backward($grad * $this->args[0]->getArray() ** $this->args[1]->getArray() * nd::log($this->args[0]->getArray()));
                break;
            case 'subtract':
                $this->args[0]->backward($grad);
                $this->args[1]->backward(-$grad);
                break;
            case 'divide':
                $this->args[0]->backward($grad / $this->args[1]->getArray());
                $this->args[1]->backward(-$grad * $this->args[0]->getArray() / $this->args[1]->getArray() ** 2);
                break;
            case 'exp':
                $this->args[0]->backward($grad * nd::exp($this->args[0]->getArray()));
                break;
            case 'tanh':
                $this->args[0]->backward($grad * (1 - (nd::tanh($this->args[0]->getArray()) ** 2)));
                break;
            case 'sqrt':
                $this->args[0]->backward($grad / (2 * nd::sqrt($this->args[0]->getArray())));
                break;
            case 'sum':
                $this->args[0]->backward(nd::ones($this->args[0]->getArray()->shape()) * $grad);
                break;
            case "log":
                $this->args[0]->backward($grad * (1 / $this->args[0]->getArray()));
                break;
            case 'sum_axis':
                // @todo Axis not supported
                $this->args[0]->backward(nd::ones($this->args[0]->getArray()->shape()) * $grad);
                break;
            case 'multiply':
                $this->args[0]->backward($grad * $this->args[1]->getArray());
                $this->args[1]->backward($this->args[0]->getArray() * $grad);
                break;
            case 'abs':
                $this->args[0]->backward($grad * nd::sign($this->args[0]->getArray()));
                break;
            case 'mean':
                $this->args[0]->backward((nd::ones($this->args[0]->getArray()->shape()) * $grad) / nd::prod(nd::array($this->args[0]->getArray()->shape())));
                break;
            case 'clip':
                $greater = nd::greater_equal($this->args[0]->getArray(), nd::ones($this->args[0]->getArray()->shape()) * $this->args[1]->getArray());
                $less = nd::less_equal($this->args[0]->getArray(), nd::ones($this->args[0]->getArray()->shape()) * $this->args[2]->getArray());
                $this->args[0]->backward($grad * $greater * $less);
                break;
            case 'dot':
                if (count($grad->shape()) == 1) {
                    $grad = nd::reshape($grad, [1, count($grad)]);
                }
                if (count($this->args[0]->getShape()) > 1 && count($this->args[0]->getShape()) > 1) {
                    $this->args[0]->backward(nd::dot($grad, nd::transpose($this->args[1]->getArray())));
                    $this->args[1]->backward(nd::dot(nd::transpose($this->args[0]->getArray()), $grad));
                } else {
                    throw new \Exception("Back propagation fatal error.");
                }
                break;
            default:
                throw new \Exception("Back propagation fatal error.");
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
}