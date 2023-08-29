<?php

namespace NumPower\Lattice\Core;

use NumPower\Lattice\Exceptions\ValueErrorException;

/**
 * Variable Operations History
 */
class Tape implements \Countable
{
    /**
     * @var ?Operation
     */
    private ?Operation $head = NULL;

    /**
     * @var
     */
    private $count;

    public function __construct() {

    }

    /**
     * @param Operation $op
     * @return void
     * @throws ValueErrorException
     */
    private function initializeHead(Operation $op): void
    {
        if (isset($this->head)) {
            throw new ValueErrorException("Tape HEAD already initialized.");
        }
        $this->count = 1;
        $this->setHead($op);
    }

    /**
     * @param Operation $op
     * @return void
     */
    private function setHead(Operation $op): void
    {
        $this->head = $op;
    }

    /**
     * @return Operation
     */
    public function getHead(): Operation
    {
        return $this->head;
    }

    /**
     * Add operation record to the tape
     *
     * @param Operation $op
     * @return void
     * @throws ValueErrorException
     */
    public function add(Operation $op): void
    {
        if (!isset($this->head)) {
            $this->initializeHead($op);
            return;
        }
        $this->count += 1;
        $op->setNext($this->getHead());
        $this->setHead($op);
    }

    /**
     * Pop last record from the tape
     *
     * @return void
     */
    public function pop(): void
    {
        array_pop($this->tape);
    }

    /**
     * @return void
     */
    public function clear(): void
    {
        unset($this->tape);
    }

    /**
     * @return int
     */
    public function count(): int
    {
        return $this->count;
    }
}