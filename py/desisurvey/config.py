"""Manage survey planning and schedule configuration data.

The normal usage is::

    >>> config = Configuration()
    >>> config.programs.BRIGHT.max_sun_altitude()
    <Quantity -13.0 deg>

Use dot notation to specify nodes in the configuration hieararchy and
function call notation to access terminal node values.

Terminal node values are first converted according to YAML rules. Strings
containing a number followed by valid astropy units are subsequently converted
to astropy quantities.  Strings of the form YYYY-MM-DD are converted to
datetime.date objects.

The configuration is implemented as a singleton so the YAML file is only
loaded and parsed the first time a Configuration() is built.  Subsequent
calls to Configuration() always return the same object.
"""
from __future__ import print_function, division

import os
import re

import yaml

import astropy.units
import astropy.utils.data


# Extract a number from a string with optional leading and
# trailing whitespace.
_float_pattern = re.compile(
    r'\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*')


class Node(object):
    """A single node of a configuration data structure.

    The purpose of this class is to allow nested dictionaries to be
    accessed using attribute dot notation, and to implement automatic
    conversion of strings of the form "<value> <units>" into corresponding
    astropy quantities.
    """
    def __init__(self, value, path=[]):
        self._path = path
        if isinstance(value, dict):
            # Remember our keys.
            self._keys = value.keys()
            # Recursively add sub-dictionaries as new child attributes.
            for name in self._keys:
                child_path = path + [name]
                self.__dict__[name] = Node(value[name], child_path)
        else:
            # Define the value of a leaf node.
            try:
                # Try to interpret as an astropy quantity with units.
                found_number = _float_pattern.match(value)
                if found_number:
                    number = float(found_number.group(1))
                    unit = value[found_number.end():]
                    try:
                        self._value = astropy.units.Quantity(number, unit)
                    except ValueError:
                        raise ValueError(
                            'Invalid unit for {0}: {1}'
                            .format('.'.join(self._path), unit))
                else:
                    self._value = value
            except TypeError:
                self._value = value

    @property
    def path(self):
        """Return the full path to this node using dot notation.
        """
        return '.'.join(self._path)

    @property
    def keys(self):
        """Return the list of keys for a non-leaf node or raise a RuntimeError
        for a terminal node.
        """
        try:
            return self._keys
        except AttributeError:
            raise RuntimeError(
                '{0} is a terminal config node.'.format(self.path))

    def __call__(self):
        """Return a terminal node's value or raise a RuntimeError for
        a non-terminal node.
        """
        try:
            return self._value
        except AttributeError:
            raise RuntimeError(
                '{0} is a non-terminal config node.'.format(self.path))


class Configuration(Node):
    """Top-level configuration data node.
    """
    __instance = None

    @staticmethod
    def reset():
        """Forget our singleton instance.  Mainly intended for unit tests."""
        Configuration.__instance = None


    def __new__(cls, file_name='config.yaml'):
        """Implement a singleton access pattern.
        """
        if Configuration.__instance is None:
            Configuration.__instance = object.__new__(cls)
            Configuration.__instance._initialize(file_name)
        elif file_name != Configuration.__instance.file_name:
            raise RuntimeError('Configuration already loaded from {0}'
                               .format(Configuration.__instance.file_name))
        return Configuration.__instance


    def __init__(self, file_name='config.yaml'):
        """Return the unique configuration object for this session.

        The configuration will be loaded from the specified file when this
        constructor is called for the first time.  Subsequent calls with
        a different file name will result in a RuntimeError.

        Parameters
        ----------
        file_name : string
            Name of a YAML file including a valid YAML extension.  The file
            is assumed to be under this package's data/ directory unless
            an absolute path is specified.
        """
        pass


    def _initialize(self, file_name):
        """Initialize a configuration data structure from a YAML file.
        """
        # Remember the file name since it is not allowed to change.
        self.file_name = file_name

        if os.path.isabs(file_name):
            full_path = file_name
        else:
            # Locate the config file in our package data/ directory.
            full_path = astropy.utils.data._find_pkg_data_path(
                os.path.join('data', file_name))

        # Validate that all mapping keys are valid python identifiers
        # and that there are no embedded sequences.
        valid_key = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*\Z')
        with open(full_path) as f:
            next_value_is_key = False
            for token in yaml.scan(f):
                if isinstance(
                    token,
                    (yaml.BlockSequenceStartToken,
                     yaml.FlowSequenceStartToken)):
                    raise RuntimeError('Config sequences not supported.')
                if next_value_is_key:
                    if not isinstance(token, yaml.ScalarToken):
                        raise RuntimeError(
                            'Invalid config key type: {0}'.format(token))
                    if not valid_key.match(token.value):
                        raise RuntimeError(
                            'Invalid config key name: {0}'.format(token.value))
                next_value_is_key = isinstance(token, yaml.KeyToken)

        # Load the config data into memory.
        with open(full_path) as f:
            Node.__init__(self, yaml.safe_load(f))

        # Output path is not set until it is first used.
        self._output_path = None


    def set_output_path(self, output_path):
        """Set the output directory for relative paths.

        The path must exist when this method is called. Used by :meth:`ge_path`.
        This method updates the configuration output_path value.

        Parameters
        ----------
        output_path : str
            A path possibly including environment variables enclosed in {...}
            that will be substituted from the current environment.

        Raises
        ------
        ValueError
            Path uses undefined environment variable or does not exist.
        """
        try:
            self._output_path = output_path.format(**os.environ)
        except KeyError as e:
            raise ValueError(
                'Environment variable not set for output_path: {0}'.format(e))
        if not os.path.isdir(self._output_path):
            raise ValueError(
                'Non-existent output_path: {0}'.format(self._output_path))
        # Update our config node.
        self.output_path._value = output_path


    def get_path(self, name):
        """Prepend this configuration's output_path to non-absolute paths.

        Configured by the ``output_path`` node and :meth:`set_output_path`.

        Parameters
        ----------
        name : str
            Absolute or relative path name, which does not need to exist yet.

        Returns
        -------
        str
            Path name to use. Relative path names will have our output_path
            prepended.  Absolute path names will be unchanged.
        """
        if self._output_path is None:
            self.set_output_path(self.output_path())
        return os.path.join(self._output_path, name)
