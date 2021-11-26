import shutil
import tempfile
import unittest
import os

from biosimulators_utils.log.data_model import Status

from biosimulators_utils.combine.io import CombineArchiveReader, CombineArchiveWriter
from biosimulators_utils.config import get_config
import biosimulators_copasi


class TestCopasiImport(unittest.TestCase):

    def setUp(self):
        self.dirname = os.path.dirname(__file__)
        self.filename = os.path.join(self.dirname, 'fixtures', 'copasi-34-export.omex')
        self.out_dir = os.path.join(self.dirname, 'out')
        self.assertTrue(os.path.exists(self.filename))

    def test_fix_archive(self):
        # ideally this shouldn't fail
        result, log = biosimulators_copasi.exec_sedml_docs_in_combine_archive(self.filename, self.out_dir)
        self.assertTrue(log.status == Status.FAILED)

        # but lets try and fix it
        archive_tmp_dir = tempfile.mkdtemp()
        config = get_config()
        config.VALIDATE_SEDML = False
        config.VALIDATE_OMEX_MANIFESTS = False

        archive = CombineArchiveReader().run(self.filename, archive_tmp_dir, config=config)
        self.assertIsNotNone(archive)

        # change the copasi export, so that biosimulators does not reject it
        biosimulators_copasi.core.fix_copasi_export(archive, archive_tmp_dir)

        # write to new archive, that should be accepted by biosimulators
        writer = CombineArchiveWriter()
        writer.run(archive, archive_tmp_dir, 'out.omex')

        results, log = biosimulators_copasi.exec_sedml_docs_in_combine_archive('out.omex', self.out_dir)
        self.assertTrue(log.status == Status.SUCCEEDED)

        # remove the archive and temp dir
        os.remove('out.omex')
        shutil.rmtree(archive_tmp_dir, True)


if __name__ == '__main__':
    unittest.main()
