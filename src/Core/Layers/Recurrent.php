<?php

namespace NumPower\Lattice\Core\Layers;

use NumPower\Lattice\Exceptions\ValueErrorException;

/**
 *  Base recurrent layer
 *
 *  Reference:
 *
 *      AkiRusProd
 *      Сustom CPU torch style machine learning framework with automatic differentiation.
 *      https://github.com/AkiRusProd/numpy-nn-model/blob/master/neunet/nn/layers/rnn.py#L52
 *
 *      Keras 1.0 - Base recurrent layer
 *      https://github.com/keras-team/keras/blob/df42e997b7d0f7c5e417c6a5a452c6ddd51e4c24/keras/layers/recurrent.py#L42
 */
abstract class Recurrent extends Layer
{
    /**
     * @var bool
     */
    private bool $returnSequences;

    /**
     * @var bool
     */
    private bool $goBackwards;

    /**
     * @var bool
     */
    private bool $stateful;

    /**
     * @var bool
     */
    private bool $unroll;

    /**
     * @var int
     */
    private int $outputDim;

    /**
     * @var ?int
     */
    private ?int $inputDim;

    /**
     * @var int|null
     */
    private ?int $inputLength;

    /**
     * @param bool $returnSequences
     * @param bool $goBackwards
     * @param bool $stateful
     * @param bool $unroll
     * @param int|null $inputDim
     * @param int|null $inputLength
     * @throws ValueErrorException
     */
    public function __construct(
        bool $returnSequences = false,
        bool $goBackwards = false,
        bool $stateful = false,
        bool $unroll = false,
        ?int $inputDim = null,
        ?int $inputLength = null
    )
    {
        $this->returnSequences = $returnSequences;
        $this->goBackwards = $goBackwards;
        $this->setStateful($stateful);
        $this->unroll = $unroll;
        $this->setInputDim($inputDim);
        $this->setInputLength($inputLength);
        if (isset($inputDim)) {
            $this->setInputShape([$inputLength, $inputDim]);
        }
        $this->setSupportMasking(true);
        $this->setInputSpec(new InputSpec(ndim: 3));
        parent::__construct("recurrent_" . substr(uniqid(), -4), true);
    }

    /**
     * @return bool
     */
    public function getStateful(): bool
    {
        return $this->stateful;
    }

    /**
     * @return void
     */
    public function setStateful(bool $stateful)
    {
        $this->stateful = $stateful;
    }

    /**
     * @return array|int[]
     */
    public function generateOutputShape(): array
    {
        if ($this->returnSequences) {
            return [$this->getInputShape()[0], $this->getInputShape()[1], $this->outputDim];
        }
        return [$this->getInputShape()[0], $this->getOutputDim()];
    }

    /**
     * @param ?int $inputDim
     * @return void
     */
    protected function setInputDim(?int $inputDim): void
    {
        $this->inputDim = $inputDim;
    }

    /**
     * @return int|null
     */
    protected function getInputDim(): ?int
    {
        return $this->inputDim;
    }

    /**
     * @param int|null $outputDim
     * @return void
     */
    protected function setOutputDim(?int $outputDim): void
    {
        $this->outputDim = $outputDim;
    }

    /**
     * @return int|null
     */
    protected function getOutputDim(): ?int
    {
        return $this->outputDim;
    }

    /**
     * @param int|null $inputLength
     * @return void
     */
    private function setInputLength(?int $inputLength): void
    {
        $this->inputLength = $inputLength;
    }

    /**
     * @return int|null
     */
    private function getInputLenth(): ?int
    {
        return $this->inputLength;
    }
}
