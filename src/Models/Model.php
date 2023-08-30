<?php

namespace NumPower\Lattice\Models;

use \NDArray as nd;
use NumPower\Lattice\Core\Layers\ILayer;
use NumPower\Lattice\Core\Layers\Layer;
use NumPower\Lattice\Core\Losses\ILoss;
use NumPower\Lattice\Core\Models\IModel;
use NumPower\Lattice\Core\Optimizers\IOptimizer;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Exceptions\ValueErrorException;
use NumPower\Lattice\Utils\EpochPrinter;
use NumPower\Lattice\Utils\LayerUtils;

class Model extends Layer implements IModel
{
    /**
     * @var int
     */
    protected int $isCompiled;

    /**
     * @var ILayer[]
     */
    protected array $layers;

    /**
     * Model Optimizer
     *
     * @var IOptimizer
     */
    private IOptimizer $optimizer;

    /**
     * @var EpochPrinter
     */
    private EpochPrinter $epochPrinter;

    /**
     * @var ILoss|null
     */
    private ?ILoss $lossFunction;

    public function __construct(?string $name = NULL) {
        $this->layers = [];
        $this->epochPrinter = new EpochPrinter();
        $this->isCompiled = false;
        ($name == NULL) ? $this->setName("model_". substr(uniqid(), -4)) : $this->setName($name);
    }

    /**
     * @return ILayer[]
     */
    public function getLayers(): array
    {
        return $this->layers;
    }

    /**
     * @param IOptimizer $optimizer
     * @param ILoss|null $loss
     * @param array|null $metrics
     * @return void
     */
    public function compile(IOptimizer $optimizer, ?ILoss $loss = NULL, ?array $metrics = NULL): void {
        $layers = $this->getLayers();
        $this->setOptimizer($optimizer);
        foreach ($this->getLayers() as $idx => $layer) {
            if ($idx == 0) {
                $layer->build([]);
                continue;
            }
            $layer->build($layers[$idx-1]->generateOutputShape());
        }
        if ($loss) {
            $this->setLossFunction($loss);
        }
        $this->isCompiled = true;
        $this->optimizer->build($this);
    }

    /**
     * @return void
     */
    public function metrics(): void {

    }

    /**
     * @param nd $y
     * @param Variable $outputs
     * @return float
     */
    public function computeLoss(\NDArray $y, Variable $outputs): Variable {
        return $this->getLossFunction()($y, $outputs);
    }

    /**
     * @param \NDArray $x
     * @param \NDArray $y
     * @return array
     */
    public function trainStep(\NDArray $x, \NDArray $y): array {
        $x_var = Variable::fromArray($x, name: "x");
        $outputs = $x_var;
        foreach ($this->getLayers() as $layer) {
            $outputs = $layer($outputs);
        }
        if (isset($this->lossFunction)) {
            $loss = $this->computeLoss($y, $outputs);
        }
        $this->getOptimizer()($loss, $this);
        return [$this->computeMetrics(), $loss, $outputs];
    }

    /**
     * @param IOptimizer $optimizer
     * @return void
     */
    public function setOptimizer(IOptimizer $optimizer): void {
        $this->optimizer = $optimizer;
    }

    /**
     * @return IOptimizer
     */
    public function getOptimizer(): IOptimizer
    {
        return $this->optimizer;
    }

    /**
     * @param ILoss $loss_fn
     * @return void
     */
    public function setLossFunction(ILoss $loss_fn): void
    {
        $this->lossFunction = $loss_fn;
    }

    /**
     * @return ILoss
     */
    public function getLossFunction(): ILoss
    {
        return $this->lossFunction;
    }

    /**
     * @param \NDArray $x
     * @param \NDArray $y
     * @param int|null $batchSize
     * @param int $epochs
     * @param float|null $validationSplit
     * @param array|null $validationData
     * @param bool|null $shuffle
     * @param callable|null $epochCallback
     * @return void
     */
    public function fit(\NDArray $x, \NDArray $y, ?int $batchSize = NULL,
                        int $epochs = 1, ?float $validationSplit = 0.0,
                        ?array $validationData = NULL, ?bool $shuffle = True,
                        ?string $epochCallback = NULL): void
    {

        for ($i = 0; $i < $epochs; $i++) {
            $this->epochPrinter->start($i, $epochs, "CPU");
            $sum_loss = Variable::fromArray(nd::array([0]));
            foreach ($x as $idx => $sample) {
                [$metrics, $loss, $outputs] = $this->trainStep(nd::reshape($sample, [1, count($sample)]), $y[$idx]);
                $this->epochPrinter->update($idx+1, count($x));
                $sum_loss = $loss->getArray();
            }
            echo "\nloss: ". ($sum_loss/count($x))[0];
            $this->epochPrinter->stop();
            if (isset($epochCallback)) {
                call_user_func($epochCallback, $i, $this, $metrics, ($sum_loss/count($x))[0], $outputs);
            }
        }
    }

    /**
     * @param $x
     * @return array
     */
    public function predict($x)
    {
        $outputs_p = [];
        foreach ($x as $idx => $sample) {
            $x_var = Variable::fromArray(nd::reshape($sample, [1, count($sample)]), name: "x");
            $outputs = $x_var;
            foreach ($this->getLayers() as $layer) {
                $outputs = $layer($outputs);
            }
            $outputs_p[] = $outputs->getArray();
        }
        return $outputs_p;
    }

    /**
     * @return void
     */
    public function summary(): void
    {
        LayerUtils::printSummary($this);
    }

    /**
     * @return array
     */
    private function computeMetrics(): array
    {
        return [];
    }

    /**
     * @param string $file_path
     * @return void
     * @throws ValueErrorException
     */
    public function save(string $file_path): void
    {
        if (file_exists($file_path)) {
            throw new ValueErrorException("File already exists.");
        }
        $file_ptr = fopen($file_path, 'wb');
        $serialized_model = serialize($this);
        fwrite($file_ptr, $serialized_model);
        fclose($file_ptr);
    }

    /**
     * @param string $file_path
     * @return Model
     */
    public static function load(string $file_path): Model
    {
        return unserialize(file_get_contents($file_path));
    }
}