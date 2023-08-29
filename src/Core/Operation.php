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
     * @param \NDArray $grad
     * @return void
     */
    public function backward(\NDArray $grad): void
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
                $this->args[0]->backward($grad * (1 - nd::tanh($this->args[0]->getArray()) ** 2));
                break;
            case 'multiply':
                $this->args[0]->backward($grad * $this->args[1]->getArray());
                $this->args[1]->backward($this->args[0]->getArray() * $grad);
                break;
            case 'dot':
                if (count($this->args[0]->getShape()) > 1 && count($this->args[0]->getShape()) > 1) {
                    $this->args[0]->backward(nd::matmul($grad, nd::transpose($this->args[1]->getArray())));
                    $this->args[1]->backward(nd::matmul(nd::transpose($this->args[0]->getArray()), $grad));
                } else {
                    throw new \Exception("Back propagation fatal error.");
                }
                break;
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