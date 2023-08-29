<?php

namespace NumPower\Lattice\Core;

class Operation
{
    /**
     * @var string
     */
    private string $name;

    /**
     * @var array
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