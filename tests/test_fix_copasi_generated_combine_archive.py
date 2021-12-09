import shutil
import tempfile
import unittest
import os
from unittest import mock

from biosimulators_utils.combine.io import CombineArchiveReader, CombineArchiveWriter
from biosimulators_utils.config import get_config
from biosimulators_utils.log.data_model import Status

import biosimulators_copasi
import biosimulators_copasi.utils


class FixCopasiGeneratedCombineArchiveTestCase(unittest.TestCase):
    def setUp(self):
        dirname = os.path.dirname(__file__)
        self.filename = os.path.join(dirname, 'fixtures', 'copasi-34-export.omex')
        self.assertTrue(os.path.exists(self.filename))
        self.archive_tmp_dir = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.archive_tmp_dir)
        shutil.rmtree(self.out_dir)

    @unittest.expectedFailure
    def test_execute_copasi_generated_archive(self):
        # ideally this shouldn't fail
        results, log = biosimulators_copasi.exec_sedml_docs_in_combine_archive(self.filename, self.out_dir)
        self.assertEqual(log.status, Status.SUCCEEDED)

    def test_executed_fixed_copasi_generated_archive(self):
        # fix the archive
        corrected_filename = os.path.join(self.archive_tmp_dir, 'archive.omex')
        biosimulators_copasi.utils.fix_copasi_generated_combine_archive(self.filename, corrected_filename)

        # check that the corrected archive can be executed
        results, log = biosimulators_copasi.exec_sedml_docs_in_combine_archive(corrected_filename, self.out_dir)
        self.assertEqual(log.status, Status.SUCCEEDED)

        results, log = biosimulators_copasi.exec_sedml_docs_in_combine_archive(
            self.filename, self.out_dir, fix_copasi_generated_combine_archive=True)
        self.assertEqual(log.status, Status.SUCCEEDED)

        with mock.patch.dict(os.environ, {'FIX_COPASI_GENERATED_COMBINE_ARCHIVE': '1'}):
            results, log = biosimulators_copasi.exec_sedml_docs_in_combine_archive(
                self.filename, self.out_dir)
        self.assertEqual(log.status, Status.SUCCEEDED)
